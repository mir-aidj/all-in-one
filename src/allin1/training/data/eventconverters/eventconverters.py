import numpy as np
import librosa
from abc import ABC
from typing import List
from numpy.typing import NDArray


class EventConverter(ABC):
  def __init__(
    self,
    times: NDArray[np.float64],
    segment_frames: int = None,
    *,
    sr: int,
    hop: int,
    start: float = None,
    end: float = None,
    # encode: bool = False,
  ):
    self.times = times
    self.segment_frames = segment_frames
    self.sr = sr
    self.hop = hop
    self.start = start
    self.end = end
  
  def get_start_end_frames(self):
    start_frame = librosa.time_to_frames(self.start, sr=self.sr, hop_length=self.hop).item() if self.start else 0
    start_frame = int(start_frame)
    if self.segment_frames:
      if self.end is not None:
        raise ValueError('total_frames and end cannot be used together')
      segment_frames = self.segment_frames
    else:
      assert self.end is not None, 'either total_frames or end should be given'
      segment_seconds = self.end - self.start
      segment_frames = round(segment_seconds * self.sr / self.hop)
    
    end_frame = start_frame + segment_frames
    
    return start_frame, end_frame
  
  def frames(
    self,
    reset_index: bool = False,
  ):
    frames = librosa.time_to_frames(self.times, sr=self.sr, hop_length=self.hop)
    start_frame, end_frame = self.get_start_end_frames()
    frames = frames[(start_frame <= frames) & (frames < end_frame)]
    
    if reset_index and self.start is not None:
      frames -= start_frame
    
    return frames
  
  def samples(
    self,
    reset_index: bool = False,
  ):
    samples = librosa.time_to_samples(self.times, sr=self.sr)
    
    if self.start is not None:
      start_sample = librosa.time_to_samples(self.start, sr=self.sr)
      samples = samples[start_sample <= samples]
    
    if self.end is not None:
      end_sample = librosa.time_to_samples(self.end, sr=self.sr)
      samples = samples[samples < end_sample]
    
    if reset_index and self.start is not None:
      samples -= start_sample
    
    return samples
  
  def of_frames(
    self,
    encode: bool = False,
  ):
    start_frame, end_frame = self.get_start_end_frames()
    num_frames = end_frame - start_frame
    
    if encode:
      event_frames = self.frames(reset_index=True)
      out = np.zeros(num_frames, dtype='float32')
      out[event_frames] = 1.
    else:
      event_frames = self.frames(reset_index=False)
      i_start, i_end = np.searchsorted(event_frames, [start_frame, end_frame])
      out = event_frames[i_start:i_end]
    
    return out


class BeatConverter(EventConverter):
  pass


class DownbeatConverter(EventConverter):
  pass


class SectionConverter(EventConverter):
  
  def __init__(
    self,
    times: NDArray[np.float64],
    section_labels: List[str],
    label_vocab: List[str],
    beat_times: NDArray[np.float64],
    segment_frames: int = None,
    *,
    sr: int,
    hop: int,
    start: float = None,
    end: float = None,
  ):
    assert len(times) + 1 == len(section_labels), 'section_labels should be one longer than section boundaries'
    super().__init__(
      times=times,
      segment_frames=segment_frames,
      sr=sr,
      hop=hop,
      start=start,
      end=end,
    )
    self.section_labels = section_labels
    self.beat_times = beat_times
    
    # Check if all_labels are unique.
    assert len(label_vocab) == len(set(label_vocab)), 'all_labels should be unique'
    self.label_vocab = label_vocab
    self.label_map = {label: i for i, label in enumerate(label_vocab)}
  
  def of_beats(self, beat_times: NDArray[np.float64] = None):
    beat_times = beat_times or self.beat_times
    names = np.array(self.section_labels)
    indices = np.searchsorted(self.times, beat_times, side='right')
    beat_sections = names[indices]
    return beat_sections
  
  def of_frames(
    self,
    encode: bool = False,
    return_labels: bool = True,
  ):
    if return_labels:
      boundary_frames = librosa.time_to_frames(self.times, sr=self.sr, hop_length=self.hop)
      start_frame, end_frame = self.get_start_end_frames()
      frame_indices = np.arange(start_frame, end_frame)
      indices = np.searchsorted(boundary_frames, frame_indices, side='right')
      labels = np.array(self.section_labels)
      if encode:
        labels = np.array([self.label_map[l] for l in labels])
      frame_labels = labels[indices]
      
      return frame_labels
    else:
      return super().of_frames(encode)
  
  @property
  def labels(self):
    assert 'end' in self.section_labels[-1]
    label_names = self.section_labels
    return np.array([self.label_map[l] for l in label_names])
