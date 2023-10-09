import numpy as np
import torch
import json

from os import PathLike
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from numpy.typing import NDArray

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
  def from_json(
    path: PathLike,
    load_activations: bool = True,
    load_embeddings: bool = True,
  ):
    from .utils import mkpath

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

    if load_activations:
      activ_path = path.with_suffix('.activ.npz')
      if activ_path.is_file():
        activs = np.load(activ_path)
        result.activations = {key: activs[key] for key in activs.files}

    if load_embeddings:
      embed_path = path.with_suffix('.embed.npy')
      if embed_path.is_file():
        result.embeddings = np.load(embed_path)

    return result


@dataclass
class AllInOnePrediction:
  raw_prob_beats: torch.FloatTensor
  raw_prob_downbeats: torch.FloatTensor
  raw_prob_sections: torch.FloatTensor
  raw_prob_functions: torch.FloatTensor

  prob_beats: torch.FloatTensor
  prob_downbeats: torch.FloatTensor
  prob_sections: torch.FloatTensor
  prob_functions: NDArray[np.float32]

  pred_beats: NDArray[np.float32]
  pred_downbeats: NDArray[np.float32]
  pred_sections: NDArray[np.float32]
  pred_functions: NDArray[np.float32]

  pred_beat_times: NDArray[np.float32]
  pred_downbeat_times: NDArray[np.float32]
  pred_section_times: NDArray[np.float32]
