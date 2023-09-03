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
  multiprocess: bool = True,
) -> Union[AnalysisResult, List[AnalysisResult]]:
  """
  Analyzes the provided audio files and returns the analysis results.

  Parameters
  ----------
  paths : Union[PathLike, List[PathLike]]
      List of paths or a single path to the audio files to be analyzed.
  out_dir : PathLike, optional
      Path to the directory where the analysis results will be saved. By default, the results will not be saved.
  visualize : Union[bool, PathLike], optional
      Whether to visualize the analysis results or not. If a path is provided, the visualizations will be saved in that
      directory. Default is False. If True, the visualizations will be saved in './viz'.
  sonify : Union[bool, PathLike], optional
      Whether to sonify the analysis results or not. If a path is provided, the sonifications will be saved in that
      directory. Default is False. If True, the sonifications will be saved in './sonif'.
  model : str, optional
      Name of the pre-trained model to be used for the analysis. Default is 'harmonix-all'. Please refer to the
      documentation for the available models.
  device : str, optional
      Device to be used for computation. Default is 'cuda' if available, otherwise 'cpu'.
  include_activations : bool, optional
      Whether to include activations in the analysis results or not.
  include_embeddings : bool, optional
      Whether to include embeddings in the analysis results or not.
  demix_dir : PathLike, optional
      Path to the directory where the source-separated audio will be saved. Default is './demix'.
  spec_dir : PathLike, optional
      Path to the directory where the spectrograms will be saved. Default is './spec'.
  keep_byproducts : bool, optional
      Whether to keep the source-separated audio and spectrograms or not. Default is False.
  multi : bool, optional
      Whether to use multiprocessing for extracting spectrograms. Default is True.

  Returns
  -------
  Union[AnalysisResult, List[AnalysisResult]]
      Analysis results for the provided audio files.
  """

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

  spec_paths = extract_spectrograms(demix_paths, spec_dir, multiprocess)

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
