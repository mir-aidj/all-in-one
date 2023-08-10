import numpy as np
import re
import json
import argparse
import torch

from os import PathLike
from typing import List
from pathlib import Path
from dataclasses import asdict
from .typings import AnalysisResult
from .analyze import analyze


def make_parser():
  cwd = Path.cwd()
  parser = argparse.ArgumentParser()
  parser.add_argument('paths', nargs='+', type=Path, default=[], help='Path to tracks')
  parser.add_argument('-a', '--activ', action='store_true',
                      help='Save frame-level raw activations from sigmoid and softmax (default: False)')
  parser.add_argument('-e', '--embed', action='store_true',
                      help='Save frame-level embeddings (default: False)')
  parser.add_argument('-o', '--out-dir', type=Path, default=cwd / './structures',
                      help='Path to a directory to store analysis results (default: ./structures)')
  parser.add_argument('-m', '--model', type=str, default='harmonix-all',
                      help='Name of the pretrained model to use (default: harmonix-all)')
  parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (default: cuda if available else cpu)')
  parser.add_argument('-k', '--keep-byproducts', action='store_true',
                      help='Keep demixed audio files and spectrograms (default: False)')
  parser.add_argument('--demix-dir', type=Path, default=cwd / 'demixed',
                      help='Path to a directory to store demixed tracks (default: ./demixed)')
  parser.add_argument('--spec-dir', type=Path, default=cwd / 'spectrograms',
                      help='Path to a directory to store spectrograms (default: ./spectrograms)')

  return parser


def main():
  parser = make_parser()
  args = parser.parse_args()

  if not args.paths:
    raise ValueError('At least one path must be specified.')

  results = analyze(
    paths=args.paths,
    model=args.model,
    device=args.device,
    include_activations=args.activ,
    include_embeddings=args.embed,
    demix_dir=args.demix_dir,
    spec_dir=args.spec_dir,
    keep_byproducts=args.keep_byproducts,
  )

  save_results(results, args.out_dir)

  print(f'=> Analysis results are successfully saved to {args.out_dir}')


def save_results(
  results: List[AnalysisResult] | AnalysisResult,
  out_dir: PathLike,
):
  if not isinstance(results, list):
    results = [results]

  out_dir = Path(out_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  for result in results:
    out_path = out_dir / result.path.with_suffix('.json').name
    result = asdict(result)
    result['path'] = str(out_path)

    activations = result.pop('activations')
    if activations is not None:
      np.savez(str(out_path.with_suffix('.activ.npz')), **activations)

    embeddings = result.pop('embeddings')
    if embeddings is not None:
      np.save(str(out_path.with_suffix('.embed.npy')), embeddings)

    json_str = json.dumps(result, indent=2)
    json_str = compact_json_number_array(json_str)
    out_path.with_suffix('.json').write_text(json_str)


def compact_json_number_array(json_str: str):
  """Compact numbers (including floats) in JSON arrays to be on the same line."""
  return re.sub(
    r'(\[\n(?:\s*\d+(\.\d+)?,\n)+\s*\d+(\.\d+)?\n\s*\])',
    lambda m: m.group(1).replace('\n', '').replace(' ', ''),
    json_str
  )


if __name__ == '__main__':
  main()
