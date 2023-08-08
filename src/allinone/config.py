from typing import List, Optional, Any
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

HARMONIX_LABELS = [
  'start',
  'end',
  'intro',
  'outro',
  'break',
  'bridge',
  'inst',
  'solo',
  'verse',
  'chorus',
]


@dataclass
class DataConfig:
  name: str

  demixed: bool
  num_instruments: int
  num_labels: int

  path_base_dir: str
  path_track_dir: str
  path_demixed_dir: str
  path_feature_dir: str
  path_no_demixed_feature_dir: str

  duration_min: float
  duration_max: float

  demucs_model: str = 'htdemucs'


@dataclass
class HarmonixConfig(DataConfig):
  name: str = 'harmonix'

  demixed: bool = True
  num_instruments: int = 4
  num_labels: int = 10

  path_base_dir: str = './data/harmonix/'
  path_track_dir: str = './data/harmonix/tracks/'
  path_demixed_dir: str = './data/harmonix/demixed/'
  path_feature_dir: str = './data/harmonix/features/'
  path_no_demixed_feature_dir: str = './data/harmonix/features_no_demixed/'
  path_metadata: str = './data/harmonix/metadata.csv'

  duration_min: int = 76
  duration_max: int = 660


defaults = [
  '_self_',
  {'data': 'harmonix'},
]


@dataclass
class Config:
  debug: bool = False
  sanity_check: bool = False
  sanity_check_size: int = 1
  offline: bool = False

  case: Optional[str] = None
  model: str = 'allinone'  # allin1, tcn

  data: DataConfig = MISSING
  defaults: List[Any] = field(default_factory=lambda: defaults)

  # Data configurations --------------------------------------------------
  sample_rate: int = 44100
  window_size: int = 2048
  num_bands: int = 12
  hop_size: int = 441  # FPS=100
  fps: int = 100
  fmin: int = 30
  fmax: int = 17000
  demucs_model: str = 'htdemucs'

  # Multi-task learning configurations ------------------------------------
  learn_rhythm: bool = True
  learn_structure: bool = True
  learn_segment: bool = True
  learn_label: bool = True

  # Training configurations -----------------------------------------------
  segment_size: Optional[float] = 300
  batch_size: int = 1

  optimizer: str = 'radam'
  sched: Optional[str] = 'plateau'
  lookahead: bool = False

  lr: float = 0.005
  warmup_lr: float = 1e-5
  warmup_epochs: int = 0
  cooldown_epochs: int = 0
  min_lr: float = 1e-7
  max_epochs: int = -1

  # Plateau scheduler.
  decay_rate: float = 0.3
  patience_epochs: int = 5
  eval_metric: str = 'val/loss'
  epochs: int = 10  # not used. just for creating the scheduler

  validation_interval_epochs: int = 3
  early_stopping_patience: int = 10

  weight_decay: float = 0.00025
  swa_lr: float = 0.15
  gradient_clip: float = 0.5

  # Model configurations --------------------------------------------------
  threshold_beat: float = 0.19
  threshold_downbeat: float = 0.19
  threshold_section: float = 0.05

  best_threshold_beat: Optional[float] = None
  best_threshold_downbeat: Optional[float] = None

  instrument_attention: bool = True
  double_attention: bool = True

  depth: int = 11
  dilation_factor: int = 2
  dilation_max: int = 3200  # 32 seconds, not in use
  num_heads: int = 2
  kernel_size: int = 5

  dim_input: int = 81
  dim_embed: int = 24
  mlp_ratio: float = 4.0
  qkv_bias: bool = True

  drop_conv: float = 0.2
  drop_path: float = 0.1
  drop_hidden: float = 0.2
  drop_attention: float = 0.2
  drop_last: float = 0.0

  act_conv: str = 'elu'
  act_transformer: str = 'gelu'

  layer_norm_eps: float = 1e-5

  # Loss configurations ---------------------------------------------------
  loss_weight_beat: float = 1.
  loss_weight_downbeat: float = 3.
  loss_weight_section: float = 15.
  loss_weight_function: float = 0.1

  # Misc ------------------------------------------------------------------
  seed: int = 1234
  fold: int = 2
  aafold: Optional[int] = None
  total_folds: int = 8

  bpm_min: int = 55
  bpm_max: int = 240
  min_hops_per_beat: int = 24  # 60 / max_bpm * sample_rate / hop_size


cs = ConfigStore.instance()
cs.store(name='config', node=Config)
cs.store(group='data', name=HarmonixConfig.name, node=HarmonixConfig)
cs.store(name=HarmonixConfig.name, node=HarmonixConfig)
