import torch

from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from ..typings import AllInOneOutput
from ..config import Config


def postprocess_metrical_structure(
  logits: AllInOneOutput,
  cfg: Config,
):
  postprocessor_downbeat = DBNDownBeatTrackingProcessor(
    beats_per_bar=[3, 4],
    threshold=cfg.best_threshold_downbeat,
    fps=cfg.fps,
  )

  raw_prob_beats = torch.sigmoid(logits.logits_beat[0])
  raw_prob_downbeats = torch.sigmoid(logits.logits_downbeat[0])

  # Transform the raw probabilities into activations indicating:
  # 1. beat but not downbeat
  # 2. downbeat
  # 3. nothing
  # for madmom's DBNDownBeatTrackingProcessor
  activations_beat = raw_prob_beats
  activations_downbeat = raw_prob_downbeats
  activations_no_beat = 1. - activations_beat
  activations_no_downbeat = 1. - activations_downbeat
  activations_no = (activations_no_beat + activations_no_downbeat) / 2.
  activations_xbeat = torch.maximum(torch.tensor(1e-8), activations_beat - activations_downbeat)
  activations_combined = torch.stack([activations_xbeat, activations_downbeat, activations_no], dim=-1)
  activations_combined /= activations_combined.sum(dim=-1, keepdim=True)
  activations_combined = activations_combined.cpu().numpy()

  pred_downbeat_times = postprocessor_downbeat(activations_combined[:, :2])

  beats = pred_downbeat_times[:, 0]
  beat_positions = pred_downbeat_times[:, 1]
  downbeats = pred_downbeat_times[beat_positions == 1., 0]

  beats = beats.tolist()
  downbeats = downbeats.tolist()
  beat_positions = beat_positions.astype('int').tolist()

  return {
    'beats': beats,
    'downbeats': downbeats,
    'beat_positions': beat_positions,
  }
