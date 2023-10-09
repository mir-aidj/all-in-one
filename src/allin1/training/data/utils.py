import numpy as np
from scipy.ndimage import maximum_filter1d


def widen_temporal_events(events, num_neighbors=2):
  """Widen temporal events by a given number of neighbors."""
  widen_events = events
  for i in range(num_neighbors):
    widen_events = maximum_filter1d(widen_events, size=3)
    neighbor_indices = np.flatnonzero((events != 1) & (widen_events > 0))
    widen_events[neighbor_indices] *= 0.5
  
  return widen_events
