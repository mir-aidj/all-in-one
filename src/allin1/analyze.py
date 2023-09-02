import numpy as np
import torch

from typing import List, Union
from tqdm import tqdm
from .demix import demix
from .spectrogram import extract_spectrograms
from .models import load_pretrained_model
from .visualize import visualize as _visualize
from .sonify import sonify as _sonify
from .postprocessing import (
  postprocess_metrical_structure,
  postprocess_functional_structure,
  estimate_tempo_from_beats,
)
from .helpers import (
  compute_activations,
  expand_paths,
  check_paths,
  rmdir_if_empty,
  save_results,
)
from .utils import mkpath
from .typings import AnalysisResult, PathLike


def analyze(
  paths: Union[PathLike, List[PathLike]],
  out_dir: PathLike = None,
  visualize: Union[bool, PathLike] = False,
  sonify: Union[bool, PathLike] = False,
  model: str = 'harmonix-all',
  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
  include_activations: bool = False,
  include_embeddings: bool = False,
  demix_dir: PathLike = './demix',
  spec_dir: PathLike = './spec',
  keep_byproducts: bool = False,
) -> Union[AnalysisResult, List[AnalysisResult]]:
  return_list = True
  if not isinstance(paths, list):
    return_list = False
    paths = [paths]
  paths = [mkpath(p) for p in paths]
  paths = expand_paths(paths)
  check_paths(paths)
  demix_dir = mkpath(demix_dir)
  spec_dir = mkpath(spec_dir)
  device = torch.device(device)
  print(f'=> Found {len(paths)} tracks to analyze.')

  demix_paths = demix(paths, demix_dir, device)

  spec_paths = extract_spectrograms(demix_paths, spec_dir)

  model = load_pretrained_model(
    model_name=model,
    device=device,
  )

  results = []
  with torch.no_grad():
    pbar = tqdm(zip(paths, spec_paths), total=len(paths))
    for path, spec_path in pbar:
      pbar.set_description(f'Analyzing {path.name}')

      spec = np.load(spec_path)
      spec = torch.from_numpy(spec).unsqueeze(0).to(device)

      logits = model(spec)

      metrical_structure = postprocess_metrical_structure(logits, model.cfg)
      functional_structure = postprocess_functional_structure(logits, model.cfg)
      bpm = estimate_tempo_from_beats(metrical_structure['beats'])

      result = AnalysisResult(
        path=path,
        bpm=bpm,
        segments=functional_structure,
        **metrical_structure,
      )
      results.append(result)

      if include_activations:
        activations = compute_activations(logits)
        result.activations = activations

      if include_embeddings:
        result.embeddings = logits.embeddings[0].cpu().numpy()

  if out_dir is not None:
    save_results(results, out_dir)

  if visualize:
    if visualize is True:
      visualize = './viz'
    _visualize(results, out_dir=visualize)
    print(f'=> Plots are successfully saved to {visualize}')

  if sonify:
    if sonify is True:
      sonify = './sonif'
    _sonify(results, out_dir=sonify)
    print(f'=> Sonified tracks are successfully saved to {sonify}')

  if not keep_byproducts:
    for path in demix_paths:
      for stem in ['bass', 'drums', 'other', 'vocals']:
        (path / f'{stem}.wav').unlink(missing_ok=True)
      rmdir_if_empty(path)
    rmdir_if_empty(demix_dir / 'htdemucs')
    rmdir_if_empty(demix_dir)

    for path in spec_paths:
      path.unlink(missing_ok=True)
    rmdir_if_empty(spec_dir)

  if not return_list:
    return results[0]
  return results
