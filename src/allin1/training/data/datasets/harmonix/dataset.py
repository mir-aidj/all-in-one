import numpy as np
import pandas as pd

from pathlib import Path
from typing import Literal, Union
from numpy.typing import NDArray
from ..datasetbase import DatasetBase
from ...utils import widen_temporal_events
from ...eventconverters import HarmonixConverter
from .....config import Config


class HarmonixDataset(DatasetBase):
  
  def __init__(
    self,
    cfg: Config,
    split: Literal['train', 'val', 'test'],
  ):
    super().__init__(cfg, split)
    
    fold = cfg.fold
    track_ids = sorted(t.stem for t in Path(cfg.data.path_track_dir).glob('*.mp3'))
    folds = np.arange(len(track_ids)) % cfg.total_folds
    
    test_fold = fold
    val_fold = (fold + 1) % cfg.total_folds
    if split == 'train':
      track_ids = [tid for tid, fold in zip(track_ids, folds) if fold not in [test_fold, val_fold]]
    elif split == 'val':
      track_ids = [tid for tid, fold in zip(track_ids, folds) if fold == val_fold]
    elif split == 'test':
      track_ids = [tid for tid, fold in zip(track_ids, folds) if fold == test_fold]
    else:
      raise ValueError(f'Unknown dataset split: {split}')
    
    df = pd.read_csv(cfg.data.path_metadata)
    df['id'] = df['File'].str.split('_').str[0]
    df = df.set_index('id')
    
    self._track_ids = track_ids
    self.feature_dir = (
      Path(cfg.data.path_feature_dir) if cfg.data.demixed
      else Path(cfg.data.path_no_demixed_feature_dir)
    )
    self.df = df
  
  @property
  def track_ids(self):
    return self._track_ids
  
  def load_features(self, track_id: str) -> NDArray:
    return np.load(self.feature_dir / f'{track_id}.npy')
  
  def create_converter(
    self,
    index: int,
    track_id: str,
    num_frames: int,
    start: float,
    end: Union[float, None],
  ) -> HarmonixConverter:
    return HarmonixConverter(
      track_id=track_id,
      total_frames=num_frames,
      sr=self.sample_rate,
      hop=self.hop,
      start=start,
      end=end,
      base_dir=self.cfg.data.path_base_dir,
    )
  
  def __getitem__(self, idx):
    data = super().__getitem__(idx)
    
    track_idx = self.track_ids[idx].split('_')[0]
    row = self.df.loc[track_idx]
    
    # Make a one-hot vector for the true tempo.
    num_bpm_units = 300
    true_bpm_int = row['BPM']
    true_bpm = np.zeros(num_bpm_units, dtype='float32')
    true_bpm[true_bpm_int] = 1.0
    widen_true_bpm = widen_temporal_events(true_bpm, num_neighbors=2)
    
    return dict(
      **data,
      true_bpm=true_bpm,
      widen_true_bpm=widen_true_bpm,
      true_bpm_int=true_bpm_int,
    )
