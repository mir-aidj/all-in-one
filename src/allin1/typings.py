import torch
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class AllInOneOutput:
  logits_beat: torch.FloatTensor = None
  logits_downbeat: torch.FloatTensor = None
  logits_section: torch.FloatTensor = None
  logits_function: torch.FloatTensor = None
  embeddings: torch.FloatTensor = None


@dataclass
class Segment:
  start: float
  end: float
  label: str


@dataclass
class AnalysisResult:
  path: Path
  bpm: int
  beats: List[float]
  downbeats: List[float]
  beat_positions: List[int]
  segments: List[Segment]
  activations: Optional[Dict[str, NDArray]] = None
  embeddings: Optional[NDArray] = None
