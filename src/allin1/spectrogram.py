import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from multiprocessing import Pool
from madmom.audio.signal import FramedSignalProcessor, Signal
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import SequentialProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor


def extract_spectrograms(demix_paths: List[Path], spec_dir: Path):
  todos = []
  spec_paths = []
  for src in demix_paths:
    dst = spec_dir / f'{src.stem}.npy'
    spec_paths.append(dst)
    if dst.is_file():
      continue
    todos.append((src, dst))

  existing = len(spec_paths) - len(todos)
  print(f'=> Found {existing} spectrograms already extracted, {len(todos)} to extract.')

  if todos:
    # Define a pre-processing chain, which is copied from madmom.
    frames = FramedSignalProcessor(
      frame_size=2048,
      fps=int(44100 / 441)
    )
    stft = ShortTimeFourierTransformProcessor()  # caching FFT window
    filt = FilteredSpectrogramProcessor(
      num_bands=12,
      fmin=30,
      fmax=17000,
      norm_filters=True
    )
    spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
    processor = SequentialProcessor([frames, stft, filt, spec])

    # Process all tracks using multiprocessing.
    with Pool() as pool:
      iterator = pool.imap(_extract_spectrogram, [
        (src, dst, processor)
        for src, dst in todos
      ])
      for _ in tqdm(iterator, total=len(todos), desc='Extracting spectrograms'):
        pass

  return spec_paths


def _extract_spectrogram(args: Tuple[Path, Path, SequentialProcessor]):
  src, dst, processor = args

  dst.parent.mkdir(parents=True, exist_ok=True)

  sig_bass = Signal(src / 'bass.wav', num_channels=1)
  sig_drums = Signal(src / 'drums.wav', num_channels=1)
  sig_other = Signal(src / 'other.wav', num_channels=1)
  sig_vocals = Signal(src / 'vocals.wav', num_channels=1)

  spec_bass = processor(sig_bass)
  spec_drums = processor(sig_drums)
  spec_others = processor(sig_other)
  spec_vocals = processor(sig_vocals)

  spec = np.stack([spec_bass, spec_drums, spec_others, spec_vocals])  # instruments, frames, bins

  np.save(str(dst), spec)
