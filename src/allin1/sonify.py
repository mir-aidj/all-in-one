import numpy as np
import librosa
import demucs.separate

from typing import Union, List, Tuple

import torch
from numpy.typing import NDArray
from .typings import AnalysisResult, PathLike, Segment
from .utils import mkpath


def sonify(
  results: Union[AnalysisResult, List[AnalysisResult]],
  out_dir: PathLike = None,
) -> Union[Tuple[NDArray, float], List[Tuple[NDArray, float]]]:
  return_list = True
  if not isinstance(results, list):
    return_list = False
    results = [results]

  sonifs = [_sonify(r) for r in results]

  if out_dir is not None:
    out_dir = mkpath(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    for r, (y, sr) in zip(results, sonifs):
      demucs.separate.save_audio(
        wav=torch.from_numpy(y),
        path=out_dir / f'{r.path.stem}.sonif{r.path.suffix}',
        samplerate=sr,
      )
      # sf.write(
      #   file=out_dir / f'{r.path.stem}.wav',
      #   data=y.T,
      #   samplerate=sr,
      # )

  if not return_list:
    return sonifs[0]
  return sonifs


def _sonify(result: AnalysisResult) -> Tuple[NDArray, float]:
  sr = 44100
  y = demucs.separate.load_track(result.path, 2, sr).numpy()
  # y, sr = librosa.load(result.path, sr=None, mono=False)

  length = y.shape[-1]
  metronome = _sonify_metronome(result, length, sr)
  boundaries = _sonify_boundaries(result.segments, length, sr)

  mixed = y + metronome + boundaries
  mixed = np.clip(mixed, -1, 1)

  return mixed, sr


def _sonify_metronome(
  result: AnalysisResult,
  length: int,
  sr: float = 44100,
):
  # Exclude downbeats from beats.
  downbeats = np.asarray(result.downbeats)
  beats = np.asarray(result.beats)
  dists = np.abs(downbeats[:, np.newaxis] - beats).min(axis=0)
  beats = beats[dists > 0.03]

  clicks_beat = librosa.clicks(
    times=beats,
    sr=sr,
    click_freq=1500,
    click_duration=0.1,
    length=length,
  )
  clicks_downbeat = librosa.clicks(
    times=downbeats,
    sr=sr,
    click_freq=3000,
    click_duration=0.1,
    length=length,
  )

  return clicks_beat + clicks_downbeat


def _sonify_boundaries(
  segments: List[Segment],
  length: int,
  sr: float = 44100,
  riser_freq_start: float = 40.0,
  riser_freq_end: float = 4000.0,
  riser_duration: float = 4.0,
  num_clicks: int = 25,
):
  click_offsets = -np.geomspace(riser_duration, 0.1, num_clicks, endpoint=False)
  boundaries = [segment.start for segment in segments if segment.label not in ['start', 'end']]
  click_times = click_offsets[:, np.newaxis] + boundaries
  click_times = click_times.T.flatten()
  click_freqs = np.geomspace(riser_freq_start, riser_freq_end, num_clicks)
  click_freqs = np.tile(click_freqs, len(boundaries))

  click_freqs = click_freqs[click_times > 0]
  click_times = click_times[click_times > 0]

  y = np.zeros((length,), dtype='float32')
  for t, f in zip(click_times, click_freqs):
    click = _synthesize_click(sr, f)
    y[int(t * sr):int(t * sr) + len(click)] += click

  for segment in segments:
    if segment.label in ['start', 'end']:
      continue
    drop = _synthesize_drop(sr)
    drop_time = segment.start
    y[int(drop_time * sr):int(drop_time * sr) + len(drop)] += drop

  return y


def _synthesize_click(
  sr: float = 44100,
  click_freq: float = 1000.0,
  click_duration: float = 0.1,
):
  angular_freq = 2 * np.pi * click_freq / float(sr)
  click = np.logspace(0, -10, num=int(sr * click_duration), base=2.0)
  click *= np.sin(angular_freq * np.arange(len(click)))
  return click


def _synthesize_drop(
  sr: float = 44100,
  drop_freq_start: float = 4000.0,
  drop_freq_end: float = 40.0,
  drop_duration: float = 0.5,
):
  freqs = np.geomspace(drop_freq_start, drop_freq_end, int(sr * drop_duration))
  drop = np.sin(2 * np.pi * freqs.cumsum() / sr)
  drop *= np.logspace(0, -4, num=int(sr * drop_duration), base=2.0)
  return drop
