import numpy as np
import torch.nn.functional as F
import torch
import librosa
from typing import Union
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from ..config import Config

def event_frames_to_time(
  tensor: Union[NDArray, torch.Tensor],
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


def median_filter_1d(tensor, filter_size=401):
  assert len(tensor.shape) in (1, 2), 'Input tensor should have 1 or 2 dimensions'
  assert filter_size % 2 == 1, 'Filter size should be an odd number'
  
  original_shape = tensor.shape
  original_dtype = tensor.dtype
  if len(original_shape) == 1:
    tensor = tensor.unsqueeze(0)
  
  # Pad the input array with the minimum value
  padding = filter_size // 2
  padded_arr = F.pad(tensor.float(), (padding, padding), mode='reflect')
  padded_arr = padded_arr.to(original_dtype)
  
  # Create a rolling window view of the padded array
  rolling_view = padded_arr.unfold(1, filter_size, 1)
  
  # Compute the median along the new dimension
  output_arr, _ = torch.median(rolling_view, dim=-1)
  
  output_arr = output_arr.reshape(original_shape)
  
  return output_arr


def local_maxima_numpy(arr, order=20):
  is_batch = len(arr.shape) == 2
  if is_batch:
    return np.stack([local_maxima_numpy(x, order) for x in arr])
  
  # Define a comparison function for argrelextrema to find local maxima
  compare_func = np.greater
  
  # Find the indices of the local maxima
  local_maxima_indices = argrelextrema(arr, compare_func, order=order)
  
  # Initialize a new numpy array with zeros and the same shape as the input array
  output_arr = np.zeros_like(arr)
  
  # Set the local maxima values in the output array
  output_arr[local_maxima_indices] = arr[local_maxima_indices]
  
  return output_arr


def binary_to_sawtooth(input_tensor: torch.FloatTensor):
  assert len(input_tensor.shape) in [1, 2], 'Input tensor must be 1D or 2D (batched 1D)'
  if len(input_tensor.shape) == 2:
    return torch.stack([binary_to_sawtooth(x) for x in input_tensor])
  
  device = input_tensor.device
  ones_indices = torch.nonzero(input_tensor).flatten()
  mean_interval = torch.diff(ones_indices).float().mean().round().int()
  pad_left = max(0, mean_interval - ones_indices[0] - 1)
  pad_right = max(0, mean_interval - (input_tensor.shape[0] - ones_indices[-1]))
  
  segment_lengths = torch.diff(
    torch.cat([
      torch.tensor([-pad_left], device=device),
      ones_indices,
      torch.tensor([input_tensor.shape[0] + pad_right], device=device),
    ])
  )
  sawtooth = [torch.linspace(0, 1, length, device=device) for length in segment_lengths]
  sawtooth = torch.cat(sawtooth)
  sawtooth = sawtooth[pad_left:]
  if pad_right > 0:
    sawtooth = sawtooth[:-pad_right]
  assert input_tensor.shape == sawtooth.shape
  return sawtooth


def quad_interp(input_tensor: torch.FloatTensor):
  assert len(input_tensor.shape) in [1, 2], 'Input tensor must be 1D or 2D (batched 1D)'
  if len(input_tensor.shape) == 2:
    return np.stack([quad_interp(x) for x in input_tensor])
  
  input_arr = input_tensor.cpu().numpy()
  arange = np.arange(len(input_arr))
  f = interp1d(arange, input_arr, kind='quadratic')
  output_arr = f(arange)
  return output_arr


def estimate_tempo_from_beats(pred_beat_times):
  beat_interval = np.diff(pred_beat_times)
  bpm = 60. / beat_interval
  # TODO: quadratic interpolation
  bpm = bpm.round()
  bincount = np.bincount(bpm.astype(int))
  bpm_range = np.arange(len(bincount))
  bpm_strength = bincount / bincount.sum()
  bpm_est = np.stack([bpm_range, bpm_strength], axis=-1)
  bpm_est = bpm_est[np.argsort(bpm_strength)[::-1]]
  bpm_est = bpm_est[bpm_est[:, 1] > 0]
  return bpm_est


def peak_picking(boundary_activation, window_past=12, window_future=6):
  # Find local maxima using a sliding window
  window_size = window_past + window_future
  assert window_size % 2 == 0, 'window_past + window_future must be even'
  window_size += 1

  # Pad boundary_activation
  boundary_activation_padded = np.pad(boundary_activation, (window_past, window_future), mode='constant')
  max_filter = sliding_window_view(boundary_activation_padded, window_size)
  local_maxima = (boundary_activation == np.max(max_filter, axis=-1)) & (boundary_activation > 0)

  # Compute strength values by subtracting the mean of the past and future windows
  past_window_filter = sliding_window_view(boundary_activation_padded[:-(window_future + 1)], window_past)
  future_window_filter = sliding_window_view(boundary_activation_padded[window_past + 1:], window_future)
  past_mean = np.mean(past_window_filter, axis=-1)
  future_mean = np.mean(future_window_filter, axis=-1)
  strength_values = boundary_activation - ((past_mean + future_mean) / 2)

  # Get boundary candidates and their corresponding strength values
  boundary_candidates = np.flatnonzero(local_maxima)
  strength_values = strength_values[boundary_candidates]

  strength_activations = np.zeros_like(boundary_activation)
  strength_activations[boundary_candidates] = strength_values

  return strength_activations
