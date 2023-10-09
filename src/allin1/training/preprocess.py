import hydra
import torch

from ..config import Config
from ..spectrogram import extract_spectrograms
from ..demix import demix
from ..helpers import expand_paths, mkpath


@hydra.main(version_base=None, config_name='config')
def main(cfg: Config):
  track_paths = expand_paths([mkpath(cfg.data.path_track_dir) / '*.mp3'])
  demix_dir = mkpath(cfg.data.path_demix_dir) / cfg.demucs_model
  feature_dir = mkpath(cfg.data.path_feature_dir)

  demix_paths = demix(
    paths=track_paths,
    demix_dir=demix_dir,
    device='cuda' if torch.cuda.is_available() else 'cpu'
  )

  spec_paths = extract_spectrograms(
    demix_paths,
    spec_dir=feature_dir,
    multiprocess=True,
  )

  print(f'Preprocessing finished. {len(spec_paths)} spectrograms saved.')


if __name__ == '__main__':
  main()
