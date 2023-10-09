import numpy as np

from abc import ABC, abstractmethod
from typing import Literal, List, Union
from numpy.typing import NDArray
from torch.utils.data import Dataset
from ..utils import widen_temporal_events
from ..eventconverters import DatasetConverter
from ....config import Config


class DatasetBase(Dataset, ABC):

  def __init__(
    self,
    cfg: Config,
    split: Literal['train', 'val', 'test', 'unseen'],
  ):
    if split not in ['train', 'val', 'test', 'unseen']:
      raise ValueError(f'Unknown dataset split: {split}')

    self.cfg = cfg
    self.split = split
    self.sample_rate = cfg.sample_rate
    self.hop = cfg.hop_size
    self.segment_size = cfg.segment_size if split == 'train' else None

  @abstractmethod
  def load_features(self, track_id: str) -> NDArray:
    pass

  @property
  @abstractmethod
  def track_ids(self) -> List[str]:
    pass

  @abstractmethod
  def create_converter(
    self,
    index: int,
    track_id: str,
    num_frames: int,
    start: float,
    end: Union[float, None],
  ) -> DatasetConverter:
    pass

  def __len__(self):
    return len(self.track_ids)

  def __getitem__(self, idx):
    track_id = self.track_ids[idx]
    spec_full = self.load_features(track_id)

    should_segment = self.segment_size is not None
    if should_segment:
      duration = spec_full.shape[1] / self.cfg.fps
      num_frames = int(self.segment_size * self.cfg.fps)
      start = (
        0 if self.cfg.sanity_check
        else np.random.uniform(0, max(0, duration - self.segment_size))
      )
      end = None
    else:
      num_frames = spec_full.shape[1]
      start = None
      end = None

    st = self.create_converter(idx, track_id, num_frames, start, end)
    start_frame, end_frame = st.beat.get_start_end_frames()
    spec = spec_full[:, start_frame:end_frame, :]

    # Normalization should be done here, but it seems madmom does it for us.

    true_beat = st.beat.of_frames(encode=True)
    true_downbeat = st.downbeat.of_frames(encode=True)
    true_section = st.section.of_frames(encode=True, return_labels=False)
    true_function = st.section.of_frames(encode=True, return_labels=True)
    true_function_list = st.section.labels

    # widen the temporal activation
    # region around the annotations to include two adjacent temporal
    # frames on either side of each quantised beat location and
    # weight them with a value of 0.5 during training.
    widen_true_beat = widen_temporal_events(true_beat, num_neighbors=1)
    widen_true_downbeat = widen_temporal_events(true_downbeat, num_neighbors=1)
    widen_true_section = widen_temporal_events(true_section, num_neighbors=2)

    true_beat_times = st.beat.times
    true_downbeat_times = st.downbeat.times
    true_section_times = st.section.times

    if should_segment:
      end_time = start + self.segment_size
      true_beat_times = true_beat_times[(start <= true_beat_times) & (true_beat_times < end_time)]
      true_downbeat_times = true_downbeat_times[(start <= true_downbeat_times) & (true_downbeat_times < end_time)]
      true_section_times = true_section_times[(start <= true_section_times) & (true_section_times < end_time)]

      true_beat_times -= start
      true_downbeat_times -= start
      true_section_times -= start

    return dict(
      track_key=track_id,
      spec=spec,

      true_beat=true_beat,
      true_downbeat=true_downbeat,
      true_section=true_section,
      true_function=true_function,

      widen_true_beat=widen_true_beat,
      widen_true_downbeat=widen_true_downbeat,
      widen_true_section=widen_true_section,

      true_beat_times=true_beat_times.tolist(),
      true_downbeat_times=true_downbeat_times.tolist(),
      true_section_times=true_section_times.tolist(),
      true_function_list=true_function_list.tolist(),
    )
