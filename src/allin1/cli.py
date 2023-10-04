import argparse
import torch

from pathlib import Path
from .analyze import analyze


def make_parser():
  cwd = Path.cwd()
  parser = argparse.ArgumentParser()
  parser.add_argument('paths', nargs='+', type=Path, default=[], help='Path to tracks')
  parser.add_argument('-o', '--out-dir', type=Path, default=cwd / './struct',
                      help='Path to a directory to store analysis results (default: ./struct)')
  parser.add_argument('-v', '--visualize', action='store_true', default=False,
                      help='Save visualizations (default: False)')
  parser.add_argument('--viz-dir', type=str, default=cwd / 'viz',
                      help='Directory to save visualizations if -v is provided (default: ./viz)')
  parser.add_argument('-s', '--sonify', action='store_true', default=False,
                      help='Save sonifications (default: False)')
  parser.add_argument('--sonif-dir', type=str, default=cwd / 'sonif',
                      help='Directory to save sonifications if -s is provided (default: ./sonif)')
  parser.add_argument('-a', '--activ', action='store_true',
                      help='Save frame-level raw activations from sigmoid and softmax (default: False)')
  parser.add_argument('-e', '--embed', action='store_true',
                      help='Save frame-level embeddings (default: False)')
  parser.add_argument('-m', '--model', type=str, default='harmonix-all',
                      help='Name of the pretrained model to use (default: harmonix-all)')
  parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (default: cuda if available else cpu)')
  parser.add_argument('-k', '--keep-byproducts', action='store_true',
                      help='Keep demixed audio files and spectrograms (default: False)')
  parser.add_argument('--demix-dir', type=Path, default=cwd / 'demix',
                      help='Path to a directory to store demixed tracks (default: ./demix)')
  parser.add_argument('--spec-dir', type=Path, default=cwd / 'spec',
                      help='Path to a directory to store spectrograms (default: ./spec)')
  parser.add_argument('--overwrite', action='store_true', default=False,
                      help='Overwrite existing files (default: False)')
  parser.add_argument('--no-multiprocess', action='store_true', default=False,
                      help='Disable multiprocessing (default: False)')

  return parser


def main():
  parser = make_parser()
  args = parser.parse_args()

  if not args.paths:
    raise ValueError('At least one path must be specified.')

  assert args.out_dir is not None, 'Output directory must be specified with --out-dir'

  analyze(
    paths=args.paths,
    out_dir=args.out_dir,
    visualize=args.viz_dir if args.visualize else False,
    sonify=args.sonif_dir if args.sonify else False,
    model=args.model,
    device=args.device,
    include_activations=args.activ,
    include_embeddings=args.embed,
    demix_dir=args.demix_dir,
    spec_dir=args.spec_dir,
    keep_byproducts=args.keep_byproducts,
    overwrite=args.overwrite,
    multiprocess=not args.no_multiprocess,
  )

  print(f'=> Analysis results are successfully saved to {args.out_dir}')


if __name__ == '__main__':
  main()
