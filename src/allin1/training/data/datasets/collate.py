import numpy as np
import torch
from collections import defaultdict


def collate_fn(raw_batch):
  variable_length_batch = defaultdict(list)
  for row in raw_batch:
    for key, value in list(row.items()):
      if isinstance(value, list):
        variable_length_batch[key].append(row.pop(key))
  
  max_T = max(x['spec'].shape[1] for x in raw_batch)
  batch = []
  for raw_data in raw_batch:
    data = {}
    for key, value in raw_data.items():
      if key in ['track_key', 'true_bpm', 'widen_true_bpm', 'true_bpm_int']:
        data[key] = value
      elif key in [
        'true_beat', 'true_downbeat', 'true_section', 'true_function',
        'widen_true_beat', 'widen_true_downbeat', 'widen_true_section',
      ]:
        data[key] = value[:max_T]
      elif key in ['spec']:
        T = raw_data[key].shape[1]
        spec = raw_data[key]
        if T < max_T:
          spec = np.pad(spec, ((0, 0), (0, max_T - T), (0, 0)), 'constant')
          mask = np.pad(np.ones(T), (0, max_T - T), 'constant')
        else:
          mask = np.ones(max_T)
        data[key] = spec
        data['mask'] = mask
      else:
        raise ValueError(f'Unknown key: {key}')
    batch.append(data)
  
  batch = torch.utils.data.default_collate(batch)
  batch = {**batch, **variable_length_batch}
  
  return batch
