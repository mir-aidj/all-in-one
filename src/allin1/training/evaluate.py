import numpy as np

import os
import wandb
import mir_eval

from pprint import pprint
from typing import List, Tuple, Dict, Mapping
from functools import partial
from tqdm import tqdm
from lightning import Trainer
from torch.multiprocessing import Pool
from madmom.evaluation.beats import BeatEvaluation

from .data import HarmonixDataModule
from .helpers import makeup_wandb_config
from .trainer import AllInOneTrainer
from .helpers import find_best_thresholds
from ..utils import mkpath
from ..config import Config, HARMONIX_LABELS
from ..typings import AllInOneOutput, AllInOnePrediction
from ..postprocessing import postprocess_functional_structure, postprocess_metrical_structure

DEBUG = False
RUN_ID = [
  'xxxxxxxx',  # Your W&B run ID here.
]
OUTDIR = 'eval/'


def main():
  global RUN_ID

  for run_id in tqdm(RUN_ID):
    print(f'=> Running evaluation of {run_id}...')
    evaluate(run_id=run_id)


def evaluate(
  run_id=None,
  model: AllInOneTrainer = None,
  trainer: Trainer = None,
):
  print('=> Evaluating...')
  if run_id:
    model, cfg, run = load_wandb_run(run_id, run_dir=OUTDIR)
    cfg.debug = DEBUG
  else:
    assert model is not None, 'Either run_id or model should be provided'
    assert trainer is not None, 'Trainer should be provided if model is provided'
    cfg = model.cfg
    run = wandb.run

  print('=> Creating data module...')
  if cfg.data.name == 'harmonix':
    dm = HarmonixDataModule(cfg)
  else:
    raise ValueError(f'Unknown dataset: {cfg.data.name}')

  if trainer is None:
    trainer = Trainer(
      # accelerator='cpu',
      devices=1,
    )

  if not cfg.debug:
    print('=> Finding best thresholds...')
    # Find the optimal thresholds.
    if (
      'best_threshold_beat' not in run.config
      or run.config['best_threshold_beat'] is None
    ):
      dm.setup('validate')
      outputs_val = trainer.predict(model, dataloaders=dm.val_dataloader())
      threshold_beat, threshold_downbeat = find_best_thresholds(outputs_val, cfg)

      if not cfg.debug:
        run.config.update({
          'best_threshold_beat': threshold_beat.item(),
          'best_threshold_downbeat': threshold_downbeat.item(),
        }, allow_val_change=True)
        if hasattr(run, 'update'):
          run.update()

    cfg.threshold_beat = run.config['best_threshold_beat']
    cfg.threshold_downbeat = run.config['best_threshold_downbeat']

  print(f'=> Evaluating with thresholds: {cfg.threshold_beat}, {cfg.threshold_downbeat}')

  scores = trainer.test(model, datamodule=dm)[0]
  if not cfg.debug:
    run.summary.update(scores)

  predict_outputs = trainer.predict(model, datamodule=dm)
  scores = compute_postprocessed_scores(predict_outputs, cfg, prefix='test/')

  print('=> Postprocessed scores on test set:')
  pprint(scores)
  if not cfg.debug:
    run.summary.update(scores)


def compute_postprocessed_scores(
  predict_outputs: List[Tuple[Dict, AllInOneOutput, AllInOnePrediction]],
  cfg: Config,
  prefix: str = '',
):
  all_scores: List[Mapping[str, float]] = []

  with Pool(os.cpu_count() // 2) as pool:
    fn = partial(
      compute_postprocessed_scores_step,
      cfg=cfg,
    )
    if cfg.debug:
      iterator = map(fn, predict_outputs)
    else:
      iterator = pool.imap(fn, predict_outputs)
    iterator = tqdm(iterator, total=len(predict_outputs), desc='Postprocessing...')

    for result in iterator:
      all_scores.append(result)

  avg_scores = {
    f'{prefix}{k}': np.mean([scores[k] for scores in all_scores])
    for k in all_scores[0].keys()
  }

  return avg_scores


def compute_postprocessed_scores_step(
  predict_output: Tuple[Dict, AllInOneOutput, AllInOnePrediction],
  cfg: Config,
) -> Mapping[str, float]:
  inputs, outputs, preds = predict_output

  pred_functional = postprocess_functional_structure(outputs, cfg)
  pred_metrical = postprocess_metrical_structure(outputs, cfg)

  eval_beat = BeatEvaluation(pred_metrical['beats'], inputs['true_beat_times'][0])
  eval_downbeat = BeatEvaluation(pred_metrical['downbeats'], inputs['true_downbeat_times'][0])

  scores_metrical = {
    'beat/f1': eval_beat.fmeasure,
    'beat/precision': eval_beat.precision,
    'beat/recall': eval_beat.recall,
    'beat/cmlt': eval_beat.cmlt,
    'beat/amlt': eval_beat.amlt,
    'downbeat/f1': eval_downbeat.fmeasure,
    'downbeat/precision': eval_downbeat.precision,
    'downbeat/recall': eval_downbeat.recall,
    'downbeat/cmlt': eval_downbeat.cmlt,
    'downbeat/amlt': eval_downbeat.amlt,
  }

  # Process ground truths.
  true_labels = inputs['true_function_list'][0]
  true_boundary_times = inputs['true_section_times'][0]
  # Some first positions are negative.
  true_boundary_times = np.maximum(true_boundary_times, 0)
  # Some last positions are out of audio length.
  duration = inputs['spec'].shape[2] * cfg.hop_size / cfg.sample_rate
  if true_boundary_times[-1] >= duration:
    true_boundary_times = true_boundary_times[:-1]
    true_labels = true_labels[:-1]
  if true_boundary_times[0] == 0:
    true_labels = true_labels[1:]  # there is no "start", beginning with "intro"
  else:
    # Else, insert "start" boundary at the beginning (0.0).
    true_boundary_times = np.insert(true_boundary_times, 0, 0.0)
  if true_boundary_times[-1] != duration:
    true_boundary_times = np.append(true_boundary_times, duration)

  pred_labels = [HARMONIX_LABELS.index(s.label) for s in pred_functional]
  pred_boundaries = np.array([[p.start, p.end] for p in pred_functional])
  true_boundaries = np.stack([true_boundary_times[:-1], true_boundary_times[1:]]).T

  scores_functional = mir_eval.segment.evaluate(
    true_boundaries, true_labels, pred_boundaries, pred_labels,
    trim=False
  )
  scores_functional = {f'segment/{k}': v for k, v in scores_functional.items()}

  scores = {**scores_functional, **scores_metrical}

  return scores


def load_wandb_run(
  run_id: str,
  run_dir: str = './wandb',
) -> Tuple[AllInOneTrainer, Config, wandb.apis.public.Run]:
  api = wandb.Api()
  run = api.run(f'taejun/danceformer/{run_id}')
  artifact = api.artifact(f'taejun/danceformer/model-{run_id}:latest', type='model')
  artifact_dir = artifact.download()
  checkpoint_path = mkpath(artifact_dir) / 'model.ckpt'
  outdir = mkpath(run_dir) / run_id
  outdir.mkdir(parents=True, exist_ok=True)
  cfg = makeup_wandb_config(run.config)
  model = AllInOneTrainer.load_from_checkpoint(
    checkpoint_path,
    map_location='cpu',
    cfg=cfg,
  )
  return model, cfg, run


if __name__ == '__main__':
  main()
