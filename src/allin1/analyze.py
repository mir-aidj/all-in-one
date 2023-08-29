import numpy as np
import torch

from os import PathLike
from typing import List, Union
from tqdm import tqdm
from .demix import demix
from .spectrogram import extract_spectrograms
from .models import load_pretrained_model
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
from .utils import _mkpath
from .typings import  AnalysisResult

def analyze(
  paths: Union[List[PathLike], PathLike],
  out_dir: PathLike = None,
  model: str = 'harmonix-all',
  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
  include_activations: bool = False,
  include_embeddings: bool = False,
  demix_dir: PathLike = './demixed',
  spec_dir: PathLike = './spectrograms',
  keep_byproducts: bool = False,
):
  # Clean up arguments.
  if not isinstance(paths, list):
    paths = [paths]
  paths = [_mkpath(p) for p in paths]
  paths = expand_paths(paths)
  check_paths(paths)
  demix_dir = _mkpath(demix_dir)
  spec_dir = _mkpath(spec_dir)
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

  if out_dir is not None:
    save_results(results, out_dir)

  if len(paths) == 1:
    return results[0]
  else:
    return results

