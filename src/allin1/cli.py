import json
import argparse
import torch

from os import PathLike
from typing import List
from pathlib import Path
from dataclasses import asdict
from .typings import AnalysisResult
from .run import analyze


def make_parser():
  cwd = Path.cwd()
  parser = argparse.ArgumentParser()
  parser.add_argument('paths', nargs='+', type=Path, default=[], help='Path to tracks')
  parser.add_argument('--out-dir', type=Path, default=cwd / './structures',
                      help='Path to a directory to store analysis results (default: ./structures)')
  parser.add_argument('--model', type=str, default='harmonix-fold0',
                      help='Name of the pretrained model to use (default: harmonix-fold0)')
  parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (default: cuda if available else cpu)')
  parser.add_argument('--demix-dir', type=Path, default=cwd / 'demixed',
                      help='Path to a directory to store demixed tracks (default: ./demixed)')
  parser.add_argument('--spec-dir', type=Path, default=cwd / 'spectrograms',
                      help='Path to a directory to store spectrograms (default: ./spectrograms)')
  # TODO: delete by-product directories
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
    demix_dir=args.demix_dir,
    spec_dir=args.spec_dir,
  )

  save_results(args.paths, results, args.out_dir)

  print(f'=> Analysis results are successfully saved to {args.out_dir}')


def save_results(
  paths: List[PathLike],
  results: List[AnalysisResult] | AnalysisResult,
  out_dir: PathLike,
):
  if not isinstance(results, list):
    results = [results]

  out_dir = Path(out_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  for path, result in zip(paths, results):
    path = out_dir / Path(path).with_suffix('.json').name
    result = asdict(result)
    with open(path.with_suffix('.json'), 'w') as f:
      json.dump(result, f, indent=2)


if __name__ == '__main__':
  main()
