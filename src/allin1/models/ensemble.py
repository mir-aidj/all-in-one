import torch
import torch.nn as nn

from typing import List
from .allinone import AllInOne
from ..typings import AllInOneOutput


class Ensemble(nn.Module):
  def __init__(self, models: List[AllInOne]):
    super().__init__()

    cfg = models[0].cfg.copy()
    cfg.best_threshold_beat = sum([model.cfg.best_threshold_beat for model in models]) / len(models)
    cfg.best_threshold_downbeat = sum([model.cfg.best_threshold_downbeat for model in models]) / len(models)

    self.cfg = cfg
    self.models = models

  def forward(self, x):
    outputs: List[AllInOneOutput] = [model(x) for model in self.models]
    avg = AllInOneOutput(
      logits_beat=torch.stack([output.logits_beat for output in outputs], dim=0).mean(dim=0),
      logits_downbeat=torch.stack([output.logits_downbeat for output in outputs], dim=0).mean(dim=0),
      logits_section=torch.stack([output.logits_section for output in outputs], dim=0).mean(dim=0),
      logits_function=torch.stack([output.logits_function for output in outputs], dim=0).mean(dim=0),
      embeddings=torch.stack([output.embeddings for output in outputs], dim=-1),
    )

    return avg
