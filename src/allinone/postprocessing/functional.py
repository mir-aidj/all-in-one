import numpy as np
import torch
from ..typings import AllInOneOutput, Segment
from ..config import Config, HARMONIX_LABELS
from .helpers import local_maxima, peak_picking, event_frames_to_time


def postprocess_functional_structure(
  logits: AllInOneOutput,
  cfg: Config,
):
  raw_prob_sections = torch.sigmoid(logits.logits_section[0])
  raw_prob_functions = torch.softmax(logits.logits_function[0], dim=0)
  prob_sections, _ = local_maxima(raw_prob_sections, filter_size=4 * cfg.min_hops_per_beat + 1)
  prob_sections = prob_sections.cpu().numpy()
  prob_functions = raw_prob_functions.cpu().numpy()

  boundary_candidates = peak_picking(
    boundary_activation=prob_sections,
    window_past=12 * cfg.fps,
    window_future=12 * cfg.fps,
  )
  boundary = boundary_candidates > 0.0

  duration = len(prob_sections) * cfg.hop_size / cfg.sample_rate
  pred_boundary_times = event_frames_to_time(boundary, cfg)
  if pred_boundary_times[0] != 0:
    pred_boundary_times = np.insert(pred_boundary_times, 0, 0)
  if pred_boundary_times[-1] != duration:
    pred_boundary_times = np.append(pred_boundary_times, duration)
  pred_boundaries = np.stack([pred_boundary_times[:-1], pred_boundary_times[1:]]).T

  #
  pred_boundary_indices = np.flatnonzero(boundary)
  pred_boundary_indices = pred_boundary_indices[pred_boundary_indices > 0]
  prob_segment_function = np.split(prob_functions, pred_boundary_indices, axis=1)
  pred_labels = [p.mean(axis=1).argmax().item() for p in prob_segment_function]

  segments = []
  for (start, end), label in zip(pred_boundaries, pred_labels):
    segment = Segment(
      start=start,
      end=end,
      label=HARMONIX_LABELS[label],
    )
    segments.append(segment)

  return segments
