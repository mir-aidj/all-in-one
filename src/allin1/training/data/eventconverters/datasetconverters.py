import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from .eventconverters import BeatConverter, DownbeatConverter, SectionConverter
from ....typings import PathLike
from ....config import HARMONIX_LABELS


class DatasetConverter(ABC):

  @property
  @abstractmethod
  def beat(self) -> BeatConverter:
    pass

  @property
  @abstractmethod
  def downbeat(self) -> DownbeatConverter:
    pass

  @property
  @abstractmethod
  def section(self) -> SectionConverter:
    pass


class HarmonixConverter(DatasetConverter):

  def __init__(
    self,
    track_id: str,
    total_frames: int = None,
    *,
    sr: int,
    hop: int,
    start: float = None,
    end: float = None,
    base_dir: PathLike = './data/harmonix/',
  ):
    self.base_dir = Path(base_dir)
    self.track_id = track_id
    self.total_frames = total_frames
    self.sr = sr
    self.hop = hop
    self.start = start
    self.end = end

    self.df_beat = pd.read_csv(
      self.base_dir / 'beats' / f'{self.track_id}.txt',
      names=['time', 'count'],
      delimiter='\t'
    )
    self.df_downbeat = self.df_beat[self.df_beat['count'] == 1]
    self.df_section = pd.read_csv(
      self.base_dir / 'segments' / f'{self.track_id}.txt',
      names=['start', 'name'],
      delimiter='\t',
    )

    section_times = self.df_section['start'].values
    section_labels = self.df_section['name'].values.tolist()
    # if section_times[0] > 0.0:
    section_labels = ['start'] + section_labels

    self._beat = BeatConverter(
      self.df_beat['time'].values,
      segment_frames=total_frames,
      sr=sr,
      hop=hop,
      start=start,
      end=end,
    )
    self._downbeat = DownbeatConverter(
      self.df_downbeat['time'].values,
      segment_frames=total_frames,
      sr=sr,
      hop=hop,
      start=start,
      end=end,
    )
    self._section = SectionConverter(
      times=section_times,
      section_labels=section_labels,
      label_vocab=HARMONIX_LABELS,
      beat_times=self.beat.times,
      segment_frames=total_frames,
      sr=sr,
      hop=hop,
      start=start,
      end=end,
    )

  @property
  def beat(self) -> BeatConverter:
    return self._beat

  @property
  def downbeat(self) -> DownbeatConverter:
    return self._downbeat

  @property
  def section(self) -> SectionConverter:
    return self._section
