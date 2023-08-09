import numpy as np
from typing import List


def estimate_tempo_from_beats(
  beats: List[float],
):
  beats = np.array(beats)
  beat_interval = np.diff(beats)
  bpm = 60. / beat_interval
  bpm = bpm.round().astype(int)
  bincount = np.bincount(bpm)
  bpm_range = np.arange(len(bincount))
  bpm_strength = bincount / bincount.sum()
  bpm_cand = np.stack([bpm_range, bpm_strength], axis=-1)
  bpm_cand = bpm_cand[np.argsort(bpm_strength)[::-1]]
  bpm_cand = bpm_cand[bpm_cand[:, 1] > 0]

  bpm_est = bpm_cand[0, 0]
  bpm_est = int(bpm_est)

  return bpm_est
