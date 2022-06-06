import pytorch_lightning as pl
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from torch_geometric.loader import DataLoader
import torch

import os, sys, inspect
currentdir = os.path.dirname( \
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

from models.hgcn import HyperGCN
from dataHandler import ParticleDataset
from lightning import LitHGCN, LitProgressBar
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger

#preparing the data
train = ParticleDataset('../../data/test','background')
#use 20% of training data for validation
train_set_size = int(len(train) * 0.8)
valid_set_size = len(train) - train_set_size
#split the train set
seed = torch.Generator().manual_seed(42)
train_set, valid_set = torch.utils.data.random_split(
    train, [train_set_size, valid_set_size], generator=seed)
train_set = DataLoader(train_set, batch_size=2, num_workers=40,drop_last=True)
valid_set = DataLoader(valid_set, batch_size=2, num_workers=40,drop_last=True)

test = ParticleDataset('../data/hz_test.hdf5','signal')
test = DataLoader(test, batch_size=4, num_workers=1)


hyper_gcn = LitHGCN(HyperGCN, 0.05)
bar = LitProgressBar()
profiler = AdvancedProfiler(dirpath='.', filename='perf_logs')
logger = TensorBoardLogger('tb_logs', name='my_model')

trainer = pl.Trainer(
        logger=logger,
        #profiler=profiler,
        #gpus=2,
        #strategy='ddp',#[DDPPlugin()],
        accelerator='cpu',
        #devices=4,
        #num_nodes=4,
        max_epochs=20,
        log_every_n_steps=2,
        callbacks=[bar],
        auto_lr_find=False)

#lr_finder = trainer.tuner.lr_find(hyper_gcn, train_set, valid_set)
#hyper_gcn.hparams.lr = lr_finder.suggestion()
#print(f'this is the best lr: {hyper_gcn.hparams.lr}')
trainer.fit(hyper_gcn, train_set, valid_set)
trainer.save_checkpoint('m_test2.ckpt')

#trainer.test(model=hgnn,dataloaders=test,ckpt_path='model.ckpt')
