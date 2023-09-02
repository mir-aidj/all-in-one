import numpy as np
import torch
import json

from os import PathLike
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from numpy.typing import NDArray
from .utils import mkpath

PathLike = Union[str, PathLike]


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

  @staticmethod
  def from_json(path: PathLike):
    path = mkpath(path)
    with open(path, 'r') as f:
      data = json.load(f)

    result = AnalysisResult(
      path=mkpath(data['path']),
      bpm=data['bpm'],
      beats=data['beats'],
      downbeats=data['downbeats'],
      beat_positions=data['beat_positions'],
      segments=[Segment(**seg) for seg in data['segments']],
    )

    activ_path = path.with_suffix('.activ.npz')
    embed_path = path.with_suffix('.embed.npy')
    if activ_path.is_file():
      activs = np.load(activ_path)
      result.activations = {key: activs[key] for key in activs.files}
    if embed_path.is_file():
      result.embeddings = np.load(embed_path)

    return result
