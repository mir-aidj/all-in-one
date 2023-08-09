import numpy as np
import torch

from os import PathLike
from pathlib import Path
from typing import List
from tqdm import tqdm
from .typings import AnalysisResult
from .demix import demix
from .spectrogram import extract_spectrograms
from .models import load_pretrained_model
from .postprocessing import (
  postprocess_metrical_structure,
  postprocess_functional_structure,
  estimate_tempo_from_beats,
)


def analyze(
  paths: PathLike | List[PathLike],
  model: str = 'harmonix-all',
  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
  demix_dir: PathLike = './demixed',
  spec_dir: PathLike = './spectrograms',
  delete_byproducts: bool = False,
):
  # Clean up arguments.
  if not isinstance(paths, list):
    paths = [paths]
  paths = [Path(p).expanduser().resolve() for p in paths]
  demix_dir = Path(demix_dir).expanduser().resolve()
  spec_dir = Path(spec_dir).expanduser().resolve()
  device = torch.device(device)

  paths = expand_paths(paths)
  check_paths(paths)
  print(f'=> Found {len(paths)} tracks to analyze.')

  demix_paths = demix(paths, demix_dir)

  spec_paths = extract_spectrograms(demix_paths, spec_dir)

  model = load_pretrained_model(
    model_name=model,
    device=device,
  )

  results = []
  with torch.no_grad():
    for spec_path in tqdm(spec_paths, desc='Analyzing'):
      spec = np.load(spec_path)
      spec = torch.from_numpy(spec).unsqueeze(0).to(device)

      logits = model(spec)

      metrical_structure = postprocess_metrical_structure(logits, model.cfg)
      functional_structure = postprocess_functional_structure(logits, model.cfg)
      bpm = estimate_tempo_from_beats(metrical_structure['beats'])

      result = AnalysisResult(**metrical_structure, bpm=bpm, segments=functional_structure)
      results.append(result)

  if delete_byproducts:
    for path in demix_paths:
      for stem in ['bass', 'drums', 'other', 'vocals']:
        (path / f'{stem}.wav').unlink(missing_ok=True)
      path.rmdir()
    for path in spec_paths:
      path.unlink()

  if len(paths) == 1:
    return results[0]
  else:
    return results


def expand_paths(paths: List[Path]):
  expanded_paths = set()
  for path in paths:
    if '*' in str(path) or '?' in str(path):
      matches = list(path.parent.glob(path.name))
      if not matches:
        raise FileNotFoundError(f'Could not find any files matching {path}')
      expanded_paths.update(matches)
    else:
      expanded_paths.add(path)

  return list(expanded_paths)


def check_paths(paths: List[Path]):
  missing_files = []
  for path in paths:
    if not path.is_file():
      missing_files.append(str(path))
  if missing_files:
    raise FileNotFoundError(f'Could not find the following files: {missing_files}')
