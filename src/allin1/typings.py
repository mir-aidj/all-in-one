import torch
from typing import List
from dataclasses import dataclass


@dataclass
class Segment:
  start: float
  end: float
  label: str


@dataclass
class AnalysisResult:
  beats: List[float]
  downbeats: List[float]
  beat_positions: List[int]
  segments: List[Segment]


@dataclass
class AllInOneOutput:
  logits_beat: torch.FloatTensor = None
  logits_downbeat: torch.FloatTensor = None
  logits_section: torch.FloatTensor = None
  logits_function: torch.FloatTensor = None
  logits_tempo: torch.FloatTensor = None
