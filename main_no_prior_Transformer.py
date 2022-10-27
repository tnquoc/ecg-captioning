import sys
import json
import pytorch_lightning as pl

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.cloud_io import load as pl_load
import numba
from numba import cuda
import torch
import matplotlib.pyplot as plt

sys.path.append('..')
print(sys.path)

# import pandas as pd

from util import get_loaders_toy_data, get_loaders
from transformer import Transformer


def cli_main(params, dev):
    pl.seed_everything(1234)
    # api_key = open("api_key.txt", "r").read()

    neptune_logger = DummyLogger()  # if dev else NeptuneLogger(api_key=api_key,

    train_loader, val_loader, test_loader, vocab = get_loaders(params, topic=False)

    model = Transformer(vocab, **params)

    checkpoint_callback = ModelCheckpoint(save_last=True)

    trainer = pl.Trainer(max_epochs=params['epochs'],
                         default_root_dir='./training/transformer/models/',
                         logger=neptune_logger,
                         log_every_n_steps=4,
                         callbacks=[checkpoint_callback],
                         gpus=1)

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)


if __name__ == '__main__':
    device = cuda.get_current_device()
    device.reset()

    params, dev = json.load(open('config_transformer.json', 'r')), False

    cli_main(params, dev)
