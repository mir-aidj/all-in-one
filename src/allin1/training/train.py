import hydra
import lightning
from lightning import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging, ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from .data import HarmonixDataModule
from .trainer import AllInOneTrainer
from .evaluate import evaluate
from ..config import Config


@hydra.main(version_base=None, config_name='config')
def main(cfg: Config):
  makeup_config(cfg)

  # Setting all the random seeds to the same value.
  # This is important in a distributed training setting.
  # Each rank will get its own set of initial weights.
  # If they don't match up, the gradients will not match either,
  # leading to training that may not converge.
  lightning.seed_everything(cfg.seed)

  if cfg.data.name == 'harmonix':
    dm = HarmonixDataModule(cfg)
  else:
    raise ValueError(f'Unknown dataset: {cfg.data.name}')

  model = AllInOneTrainer(cfg)

  wandb_logger = WandbLogger(
    project='danceformer',
    tags=[
           f'fold{cfg.fold}'
         ] + (
           [cfg.case] if cfg.case else []
         ),
    log_model=False if cfg.debug or cfg.sanity_check or cfg.offline else 'all',
    offline=cfg.debug or cfg.sanity_check or cfg.offline,
  )
  wandb_logger.log_hyperparams(cfg)
  wandb_logger.experiment.define_metric('val/loss', summary='min')

  callbacks = [
    ModelCheckpoint(monitor='val/loss', mode='min'),
    EarlyStopping(
      monitor='val/loss',
      mode='min',
      patience=cfg.early_stopping_patience,
      min_delta=1e-4,
      log_rank_zero_only=True,
      verbose=True
    ),
    LearningRateMonitor(),
  ]
  if cfg.swa_lr > 1e-4:
    callbacks.append(StochasticWeightAveraging(swa_lrs=cfg.swa_lr))

  trainer = Trainer(
    accelerator='cpu' if cfg.debug else 'auto',
    # For some reason, ddp stucks at evaluation...
    # devices=1 if cfg.sanity_check or cfg.debug else 'auto',
    devices=1,
    gradient_clip_val=cfg.gradient_clip,
    logger=wandb_logger,
    callbacks=None if cfg.sanity_check else callbacks,
    check_val_every_n_epoch=cfg.validation_interval_epochs,
    max_epochs=cfg.max_epochs,
    fast_dev_run=cfg.debug and not cfg.sanity_check,
    overfit_batches=cfg.sanity_check_size if cfg.sanity_check else 0,
  )
  if cfg.sanity_check:
    trainer.limit_val_batches = 0

  if trainer.is_global_zero:
    print('=' * 80)
    print('Config')
    print('=' * 80)
    print(OmegaConf.to_yaml(cfg))
    print('=' * 80)

  trainer.fit(
    model=model,
    datamodule=dm,
  )
  print(f'=> Finished training.')

  if trainer.is_global_zero:
    print('=> Running evaluation...')
    evaluate(
      model=model,
      trainer=trainer,
    )


def makeup_config(cfg: Config):
  if cfg.sanity_check:
    cfg.sched = None
    cfg.warmup_epochs = 0
    cfg.weight_decay = 0
    cfg.drop_conv = 0
    cfg.drop_path = 0
    cfg.drop_hidden = 0
    cfg.drop_attention = 0
    cfg.validation_interval_epochs = 50


if __name__ == '__main__':
  main()
