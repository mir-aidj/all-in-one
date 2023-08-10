import numpy as np
import torch

from os import PathLike
from pathlib import Path
from typing import List
from tqdm import tqdm
from .typings import AllInOneOutput, AnalysisResult
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
  include_activations: bool = False,
  include_embeddings: bool = False,
  demix_dir: PathLike = './demixed',
  spec_dir: PathLike = './spectrograms',
  keep_byproducts: bool = False,
):
  # Clean up arguments.
  if not isinstance(paths, list):
    paths = [paths]
  paths = [Path(p).expanduser().resolve() for p in paths]
  paths = expand_paths(paths)
  check_paths(paths)
  demix_dir = Path(demix_dir).expanduser().resolve()
  spec_dir = Path(spec_dir).expanduser().resolve()
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

  if len(paths) == 1:
    return results[0]
  else:
    return results


def compute_activations(logits: AllInOneOutput):
  activations_beat = torch.sigmoid(logits.logits_beat[0]).cpu().numpy()
  activations_downbeat = torch.sigmoid(logits.logits_downbeat[0]).cpu().numpy()
  activations_segment = torch.sigmoid(logits.logits_section[0]).cpu().numpy()
  activations_label = torch.softmax(logits.logits_function[0], dim=0).cpu().numpy()
  return {
    'beat': activations_beat,
    'downbeat': activations_downbeat,
    'segment': activations_segment,
    'label': activations_label,
  }


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


def rmdir_if_empty(path: Path):
  try:
    path.rmdir()
  except (FileNotFoundError, OSError):
    pass
