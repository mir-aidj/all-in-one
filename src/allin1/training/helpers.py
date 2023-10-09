import numpy as np
import torch.nn.functional as F

import ast
import torch
import librosa

from typing import Union
from tqdm import tqdm
from omegaconf import DictConfig
from numpy.typing import NDArray
from madmom.evaluation.beats import BeatEvaluation, BeatMeanEvaluation
from ..config import Config
from ..typings import AllInOnePrediction


def makeup_wandb_config(run_config):
  run_config = run_config.copy()
  if 'data' in run_config:
    run_config['data'] = ast.literal_eval(run_config['data'])
  elif 'data/name' in run_config:
    run_config = unflatten_config(run_config, delimiter='/')
  return DictConfig(run_config)


def unflatten_config(d, delimiter='/'):
  unflattened = unflatten_dict(d, delimiter=delimiter)
  return DictConfig(unflattened)


def unflatten_dict(d, delimiter='/'):
  unflattened = {}
  for key, value in d.items():
    keys = key.split(delimiter)
    current_level = unflattened
    for k in keys[:-1]:
      current_level = current_level.setdefault(k, {})
    current_level[keys[-1]] = value
  return unflattened


def event_frames_to_time(
  tensor: Union[torch.Tensor, NDArray],
  cfg: Config = None,
  sample_rate: int = None,
  hop_size: int = None,
):
  """
  Args:
    tensor: a binary event tensor with shape (batch, frame)
  """
  assert len(tensor.shape) in (1, 2), 'Input tensor should have 1 or 2 dimensions'

  if cfg is not None:
    sample_rate = cfg.sample_rate
    hop_size = cfg.hop_size

  if torch.is_tensor(tensor):
    tensor = tensor.cpu().numpy()

  original_shape = tensor.shape
  if len(original_shape) == 1:
    tensor = tensor[None, :]

  batch_size = tensor.shape[0]
  i_examples, i_frames = np.where(tensor)
  times = librosa.frames_to_time(i_frames, sr=sample_rate, hop_length=hop_size)
  times = [times[i_examples == i] for i in range(batch_size)]

  if len(original_shape) == 1:
    times = times[0]
  return times


def local_maxima(tensor, filter_size=41):
  assert len(tensor.shape) in (1, 2), 'Input tensor should have 1 or 2 dimensions'
  assert filter_size % 2 == 1, 'Filter size should be an odd number'

  original_shape = tensor.shape
  if len(original_shape) == 1:
    tensor = tensor.unsqueeze(0)

  # Pad the input array with the minimum value
  padding = filter_size // 2
  padded_arr = F.pad(tensor, (padding, padding), mode='constant', value=-torch.inf)

  # Create a rolling window view of the padded array
  rolling_view = padded_arr.unfold(1, filter_size, 1)

  # Find the indices of the local maxima
  center = filter_size // 2
  local_maxima_mask = torch.eq(rolling_view[:, :, center], torch.max(rolling_view, dim=-1).values)
  local_maxima_indices = local_maxima_mask.nonzero()

  # Initialize a new PyTorch tensor with zeros and the same shape as the input tensor
  output_arr = torch.zeros_like(tensor)

  # Set the local maxima values in the output tensor
  output_arr[local_maxima_mask] = tensor[local_maxima_mask]

  output_arr = output_arr.reshape(original_shape)

  return output_arr, local_maxima_indices


def find_best_thresholds(predict_outputs, cfg: Config):
  probs_beat, trues_beat = [], []
  probs_downbeat, trues_downbeat = [], []
  for inputs, outputs, preds in predict_outputs:
    preds: AllInOnePrediction
    probs_beat.append(preds.raw_prob_beats[0])
    probs_downbeat.append(preds.raw_prob_downbeats[0])
    trues_beat.append(inputs['true_beat_times'][0])
    trues_downbeat.append(inputs['true_downbeat_times'][0])

  print('=> Start finding best thresholds for beat and downbeat...')
  threshold_beat, _ = find_best_threshold(probs_beat, trues_beat, cfg, cfg.min_hops_per_beat + 1)
  threshold_downbeat, _ = find_best_threshold(probs_downbeat, trues_downbeat, cfg, 4 * cfg.min_hops_per_beat + 1)

  return threshold_beat, threshold_downbeat


def find_best_threshold(probs, trues, cfg: Config, filter_size: int):
  results = {}
  for threshold in tqdm(
    np.linspace(0, 0.5, 51),
    desc='Finding best threshold...',
  ):
    results[threshold] = []
    for prob, true in zip(probs, trues):
      lmprob_beats, _ = local_maxima(prob, filter_size=filter_size)
      pred_beats = (lmprob_beats > threshold).numpy()
      pred_beat_times = event_frames_to_time(pred_beats, cfg)
      eval = BeatEvaluation(pred_beat_times, true)
      results[threshold].append(eval)
    results[threshold] = BeatMeanEvaluation(results[threshold])

  best_threshold = max(results, key=lambda k: results[k].fmeasure)
  return best_threshold, results[best_threshold]
