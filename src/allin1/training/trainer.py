import warnings
import librosa
import numpy as np
import torch.nn.functional as F
import torch

from typing import Dict, Union
from lightning import LightningModule
from madmom.evaluation.beats import BeatEvaluation, BeatMeanEvaluation
from numpy.typing import NDArray
from sklearn.metrics import f1_score, accuracy_score
from timm.optim.optim_factory import create_optimizer_v2 as create_optimizer
from timm.scheduler import create_scheduler
from timm.scheduler.scheduler import Scheduler

from ..models import AllInOne
from ..typings import AllInOneOutput, AllInOnePrediction
from ..config import Config
from .helpers import local_maxima

# For ignoring following warnings from madmom.evaluation
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=UserWarning, message='Not enough beat annotations')
warnings.filterwarnings('ignore', category=UserWarning, message='The epoch parameter')
warnings.filterwarnings('ignore', category=UserWarning, message='no annotated tempo strengths given')


class AllInOneTrainer(LightningModule):
  scheduler: Scheduler

  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg

    if cfg.model == 'allinone':
      self.model = AllInOne(cfg)
    else:
      raise NotImplementedError(f'Unknown model: {cfg.model}')

    self.lr = cfg.lr

  def forward(self, x):
    return self.model(x)

  def configure_optimizers(self):
    optimizer = create_optimizer(
      self,
      opt=self.cfg.optimizer,
      lr=self.cfg.lr,
      weight_decay=self.cfg.weight_decay,
    )
    if self.cfg.sched is not None:
      self.scheduler, _ = create_scheduler(self.cfg, optimizer)

    return {
      'optimizer': optimizer,
    }

  def on_train_epoch_end(self) -> None:
    if self.cfg.sanity_check:
      return

    if self.cfg.sched == 'plateau':
      if (self.current_epoch + 1) % self.cfg.validation_interval_epochs == 0:
        optimizer = self.trainer.optimizers[0]
        old_lr = optimizer.param_groups[0]['lr']

        metric = self.trainer.callback_metrics[self.cfg.eval_metric]
        self.scheduler.step(epoch=self.current_epoch + 1, metric=metric)

        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
          print(f'=> The LR is decayed from {old_lr} to {new_lr}. '
                f'Loading the best model: {self.cfg.eval_metric}={self.trainer.checkpoint_callback.best_model_score}')
          self.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path, cfg=self.cfg)
      elif self.current_epoch + 1 <= self.cfg.warmup_epochs:
        self.scheduler.step(epoch=self.current_epoch + 1)
    else:
      self.scheduler.step(epoch=self.current_epoch + 1)

  def training_step(self, batch, batch_idx):
    batch_size = batch['spec'].shape[0]
    outputs: AllInOneOutput = self(batch['spec'])
    losses = self.compute_losses(outputs, batch, prefix='train/')
    loss = losses.pop('train/loss')
    self.log('train/loss', loss, prog_bar=True, batch_size=batch_size)
    self.log_dict(losses, batch_size=batch_size)

    if (self.current_epoch + 1) % self.cfg.validation_interval_epochs == 0 or self.cfg.debug:
      predictions = self.compute_predictions(outputs, mask=batch['mask'])
      scores = self.compute_metrics(predictions, batch, prefix='train/')
      self.log_dict(scores, sync_dist=True, on_epoch=True, batch_size=batch_size)

      if self.cfg.sanity_check:
        print('\n')
        for k, v in {**losses, **scores}.items():
          print(k, v.item())
        print('\n')

    return loss

  def evaluation_step(self, batch, batch_idx, prefix=None):
    batch_size = batch['spec'].shape[0]
    outputs: AllInOneOutput = self(batch['spec'])
    losses = self.compute_losses(outputs, batch, prefix)
    predictions = self.compute_predictions(outputs)
    scores = self.compute_metrics(predictions, batch, prefix)
    self.log_dict(losses, sync_dist=True, batch_size=batch_size)
    self.log_dict(scores, sync_dist=True, batch_size=batch_size)

  def validation_step(self, batch, batch_idx):
    self.evaluation_step(batch, batch_idx, prefix='val/')

  def test_step(self, batch, batch_idx):
    self.evaluation_step(batch, batch_idx, prefix='test/')

  def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
    assert batch['spec'].shape[0] == 1, 'Batch size must be 1 for prediction'
    outputs: AllInOneOutput = self(batch['spec'])
    # losses = self.compute_losses(outputs, batch)
    predictions = self.compute_predictions(outputs)
    # scores = self.compute_metrics(predictions, batch)
    return batch, outputs, predictions

  def compute_losses(self, outputs: AllInOneOutput, batch: Dict, prefix: str = None):
    loss = 0.0
    losses = {}

    loss_beat = F.binary_cross_entropy_with_logits(
      outputs.logits_beat, batch['widen_true_beat'],
      reduction='none',
    )
    loss_downbeat = F.binary_cross_entropy_with_logits(
      outputs.logits_downbeat, batch['widen_true_downbeat'],
      reduction='none',
    )
    loss_section = F.binary_cross_entropy_with_logits(
      outputs.logits_section, batch['widen_true_section'],
      reduction='none',
    )
    loss_function = F.cross_entropy(
      outputs.logits_function, batch['true_function'],
      reduction='none',
    )

    loss_beat = torch.mean(batch['mask'] * loss_beat)
    loss_downbeat = torch.mean(batch['mask'] * loss_downbeat)
    loss_section = torch.mean(batch['mask'] * loss_section)
    loss_function = torch.mean(batch['mask'] * loss_function)

    loss_beat *= self.cfg.loss_weight_beat
    loss_downbeat *= self.cfg.loss_weight_downbeat
    loss_section *= self.cfg.loss_weight_section
    loss_function *= self.cfg.loss_weight_function

    if self.cfg.learn_rhythm:
      loss += loss_beat + loss_downbeat
    if self.cfg.learn_structure:
      if self.cfg.learn_label:
        loss += loss_function
      if self.cfg.learn_segment:
        loss += loss_section

    losses.update(
      loss=loss,
      loss_beat=loss_beat,
      loss_downbeat=loss_downbeat,
      loss_section=loss_section,
      loss_function=loss_function,
    )
    if prefix:
      losses = prefix_dict(losses, prefix)
    return losses

  def compute_predictions(self, outputs: AllInOneOutput, mask=None):
    raw_prob_beats = torch.sigmoid(outputs.logits_beat.detach())
    raw_prob_downbeats = torch.sigmoid(outputs.logits_downbeat.detach())
    raw_prob_sections = torch.sigmoid(outputs.logits_section.detach())
    raw_prob_functions = torch.softmax(outputs.logits_function.detach(), dim=1)

    prob_beats, _ = local_maxima(raw_prob_beats, filter_size=self.cfg.min_hops_per_beat + 1)
    prob_downbeats, _ = local_maxima(raw_prob_downbeats, filter_size=4 * self.cfg.min_hops_per_beat + 1)
    prob_sections, _ = local_maxima(raw_prob_sections, filter_size=4 * self.cfg.min_hops_per_beat + 1)
    prob_functions = raw_prob_functions.cpu().numpy()

    if mask is not None:
      prob_beats *= mask
      prob_downbeats *= mask
      prob_sections *= mask

    pred_beats = prob_beats > self.cfg.threshold_beat
    pred_downbeats = prob_downbeats > self.cfg.threshold_downbeat
    pred_sections = prob_sections > self.cfg.threshold_section
    pred_functions = np.argmax(prob_functions, axis=1)
    if mask is not None:
      pred_functions = np.where(mask.cpu().numpy(), pred_functions, -1)

    pred_beat_times = self.tensor_to_time(pred_beats)
    pred_downbeat_times = self.tensor_to_time(pred_downbeats)
    pred_section_times = self.tensor_to_time(pred_sections)

    p = AllInOnePrediction(
      raw_prob_beats=raw_prob_beats,
      raw_prob_downbeats=raw_prob_downbeats,
      raw_prob_sections=raw_prob_sections,
      raw_prob_functions=raw_prob_functions,

      prob_beats=prob_beats,
      prob_downbeats=prob_downbeats,
      prob_sections=prob_sections,
      prob_functions=prob_functions,

      pred_beats=pred_beats,
      pred_downbeats=pred_downbeats,
      pred_sections=pred_sections,
      pred_functions=pred_functions,

      pred_beat_times=pred_beat_times,
      pred_downbeat_times=pred_downbeat_times,
      pred_section_times=pred_section_times,
    )

    return p

  def compute_metrics(self, p: AllInOnePrediction, batch: Dict, prefix: str = None):
    eval_beat = [
      BeatEvaluation(pred, true)
      for pred, true in zip(p.pred_beat_times, batch['true_beat_times'])
      if len(pred) > 1 and len(true) > 1
    ]
    eval_downbeat = [
      BeatEvaluation(pred, true)
      for pred, true in zip(p.pred_downbeat_times, batch['true_downbeat_times'])
      if len(pred) > 1 and len(true) > 1
    ]
    eval_section = [
      BeatEvaluation(pred, true, fmeasure_window=0.5)
      for pred, true in zip(p.pred_section_times, batch['true_section_times'])
      if len(pred) > 1 and len(true) > 1
    ]

    score_beat = BeatMeanEvaluation(eval_beat)
    score_downbeat = BeatMeanEvaluation(eval_downbeat)
    score_section = BeatMeanEvaluation(eval_section)

    mask = batch['mask'].cpu().numpy()
    true_functions = batch['true_function'].cpu().numpy()
    true_functions = np.where(mask, true_functions, -1)
    true_functions = true_functions.flatten()
    true_functions = true_functions[true_functions != -1]

    pred_functions = p.pred_functions
    pred_functions = np.where(mask, pred_functions, -1)
    pred_functions = pred_functions.flatten()
    pred_functions = pred_functions[pred_functions != -1]

    function_f1 = f1_score(true_functions, pred_functions, average='macro')
    function_accuracy = accuracy_score(true_functions, pred_functions)

    d = dict(
      beat_f1=score_beat.fmeasure,
      beat_precision=score_beat.precision,
      beat_recall=score_beat.recall,
      beat_cmlt=score_beat.cmlt,
      beat_amlt=score_beat.amlt,
      downbeat_f1=score_downbeat.fmeasure,
      downbeat_precision=score_downbeat.precision,
      downbeat_recall=score_downbeat.recall,
      downbeat_cmlt=score_downbeat.cmlt,
      downbeat_amlt=score_downbeat.amlt,
      section_f1=score_section.fmeasure,
      section_precision=score_section.precision,
      section_recall=score_section.recall,
      function_f1=function_f1,
      function_acc=function_accuracy,
    )

    if prefix:
      d = prefix_dict(d, prefix)
    return d

  def tensor_to_time(self, tensor: Union[torch.Tensor, NDArray]):
    """
    Args:
      tensor: a binary event tensor with shape (batch, frame)
    """
    if torch.is_tensor(tensor):
      tensor = tensor.cpu().numpy()
    batch_size = tensor.shape[0]
    i_examples, i_frames = np.where(tensor)
    times = librosa.frames_to_time(i_frames, sr=self.cfg.sample_rate, hop_length=self.cfg.hop_size)
    times = [times[i_examples == i] for i in range(batch_size)]
    return times

  def on_fit_end(self):
    print('=> Fit ended.')
    if self.trainer.is_global_zero and self.trainer.checkpoint_callback.best_model_path:
      print('=> Loading best model...')
      self.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path, cfg=self.cfg)
      print('=> Loaded best model.')


def prefix_dict(d: Dict, prefix: str):
  return {
    prefix + key: value
    for key, value in d.items()
  }
