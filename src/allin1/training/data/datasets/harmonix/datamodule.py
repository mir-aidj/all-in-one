from lightning import LightningDataModule
from torch.utils.data import DataLoader
from .dataset import HarmonixDataset
from ..collate import collate_fn
from .....config import Config


class HarmonixDataModule(LightningDataModule):
  dataset_train: HarmonixDataset
  dataset_val: HarmonixDataset
  dataset_test: HarmonixDataset
  
  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg
  
  def setup(self, stage: str):
    if stage == 'fit':
      self.dataset_train = HarmonixDataset(self.cfg, split='train')
      # if self.cfg.debug:
      #   self.dataset_train = Subset(self.dataset_train, range(1))
    
    if stage in ['fit', 'validate']:
      if self.cfg.sanity_check:
        self.dataset_val = self.dataset_train
      else:
        self.dataset_val = HarmonixDataset(self.cfg, split='val')
      # if self.cfg.debug:
      #   self.dataset_val = Subset(self.dataset_val, range(1))
    
    if stage in ['test', 'predict']:
      if self.cfg.sanity_check:
        self.dataset_test = self.dataset_train
      else:
        self.dataset_test = HarmonixDataset(self.cfg, split='test')
      # if self.cfg.debug:
      #   self.dataset_test = Subset(self.dataset_test, range(1))
  
  def train_dataloader(self):
    return DataLoader(
      self.dataset_train,
      batch_size=self.cfg.batch_size,
      shuffle=True,
      num_workers=0 if self.cfg.debug else 2,
      collate_fn=collate_fn,
    )
  
  def val_dataloader(self):
    return DataLoader(
      self.dataset_val,
      batch_size=1,
      shuffle=False,
      num_workers=0 if self.cfg.debug else 1,
      collate_fn=collate_fn,
    )
  
  def test_dataloader(self):
    return DataLoader(
      self.dataset_test,
      batch_size=1,
      shuffle=False,
      num_workers=0 if self.cfg.debug else 1,
      collate_fn=collate_fn,
    )
  
  def predict_dataloader(self):
    return self.test_dataloader()
