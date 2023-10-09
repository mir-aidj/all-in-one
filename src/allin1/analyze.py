import torch

from typing import List, Union
from tqdm import tqdm
from .demix import demix
from .spectrogram import extract_spectrograms
from .models import load_pretrained_model
from .visualize import visualize as _visualize
from .sonify import sonify as _sonify
from .helpers import (
  run_inference,
  expand_paths,
  check_paths,
  rmdir_if_empty,
  save_results,
)
from .utils import mkpath, load_result
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
  overwrite: bool = False,
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
  overwrite : bool, optional
      Whether to overwrite the existing analysis results or not. Default is False.
  multiprocess : bool, optional
      Whether to use multiprocessing for spectrogram extraction, visualization, and sonification. Default is True.

  Returns
  -------
  Union[AnalysisResult, List[AnalysisResult]]
      Analysis results for the provided audio files.
  """

  # Clean up the arguments.
  return_list = True
  if not isinstance(paths, list):
    return_list = False
    paths = [paths]
  if not paths:
    raise ValueError('At least one path must be specified.')
  paths = [mkpath(p) for p in paths]
  paths = expand_paths(paths)
  check_paths(paths)
  demix_dir = mkpath(demix_dir)
  spec_dir = mkpath(spec_dir)

  # Check if the results are already computed.
  if out_dir is None or overwrite:
    todo_paths = paths
    exist_paths = []
  else:
    out_paths = [mkpath(out_dir) / path.with_suffix('.json').name for path in paths]
    todo_paths = [path for path, out_path in zip(paths, out_paths) if not out_path.exists()]
    exist_paths = [out_path for path, out_path in zip(paths, out_paths) if out_path.exists()]

  print(f'=> Found {len(exist_paths)} tracks already analyzed and {len(todo_paths)} tracks to analyze.')
  if exist_paths:
    print(f'=> To re-analyze, please use --overwrite option.')

  # Load the results for the tracks that are already analyzed.
  results = []
  if exist_paths:
    results += [
      load_result(
        exist_path,
        load_activations=include_activations,
        load_embeddings=include_embeddings,
      )
      for exist_path in tqdm(exist_paths, desc='Loading existing results')
    ]

  # Analyze the tracks that are not analyzed yet.
  if todo_paths:
    # Run HTDemucs for source separation only for the tracks that are not analyzed yet.
    demix_paths = demix(todo_paths, demix_dir, device)

    # Extract spectrograms for the tracks that are not analyzed yet.
    spec_paths = extract_spectrograms(demix_paths, spec_dir, multiprocess)

    # Load the model.
    model = load_pretrained_model(
      model_name=model,
      device=device,
    )

    with torch.no_grad():
      pbar = tqdm(zip(todo_paths, spec_paths), total=len(todo_paths))
      for path, spec_path in pbar:
        pbar.set_description(f'Analyzing {path.name}')

        result = run_inference(
          path=path,
          spec_path=spec_path,
          model=model,
          device=device,
          include_activations=include_activations,
          include_embeddings=include_embeddings,
        )

        # Save the result right after the inference.
        # Checkpointing is always important for this kind of long-running tasks...
        # for my mental health...
        if out_dir is not None:
          save_results(result, out_dir)

        results.append(result)

  # Sort the results by the original order of the tracks.
  results = sorted(results, key=lambda result: paths.index(result.path))

  if visualize:
    if visualize is True:
      visualize = './viz'
    _visualize(results, out_dir=visualize, multiprocess=multiprocess)
    print(f'=> Plots are successfully saved to {visualize}')

  if sonify:
    if sonify is True:
      sonify = './sonif'
    _sonify(results, out_dir=sonify, multiprocess=multiprocess)
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
