from data.data_frame import Frame_Dataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DatasetModule(pl.LightningDataModule):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.dataset
        self.dataset_name = cfg.dataset['name']

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_set = None
        self.valid_set = None
        self.test_set = None

        if self.dataset_name == 'frame':
            if stage == 'test':
                self.test_set = Frame_Dataset(basedir=self.cfg['basedir'], mode='test')
            else:
                self.train_set = Frame_Dataset(basedir=self.cfg['basedir'], mode='train', data_size=self.cfg['data_size'])
                self.valid_set = Frame_Dataset(basedir=self.cfg['basedir'], mode='val')
       

    
    def train_dataloader(self):
        return DataLoader(self.train_set,
                          num_workers=self.cfg['num_workers'],
                          batch_size=self.cfg['batch_size'],
                          shuffle=True, 
                          drop_last=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_set,
                          num_workers=self.cfg['num_workers'],
                          batch_size=self.cfg['batch_size'],
                          shuffle=False,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          num_workers=self.cfg['num_workers'],
                          batch_size=self.cfg['batch_size'],
                          shuffle=False,
                          drop_last=True)