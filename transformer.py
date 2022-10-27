import torch
import random
from torch import nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import collections
from ecgresnet import ECGResNet
from transformer_network import TransformerModule
from utils_model import get_next_word

import copy
import matplotlib.pyplot as plt


class Transformer(pl.LightningModule):
    def __init__(self, vocab, in_length, in_channels,
                 n_grps, N, num_classes,
                 dropout, first_width,
                 stride, dilation, num_layers, d_mode, nhead, epochs, **kwargs):

        super().__init__()

        self.vocab_length = len(vocab)
        self.vocab = vocab
        self.save_hyperparameters()
        self.model = ECGResNet(in_length, in_channels,
                               n_grps, N, num_classes,
                               dropout, first_width,
                               stride, dilation)

        self.model.flatten = Identity()
        self.model.fc1 = AveragePool()
        self.model.fc2 = AveragePool()

        self.pre_train = False
        self.feature_embedding = nn.Linear(512, d_mode)
        self.embed = nn.Embedding(len(vocab), d_mode)
        self.transformer = TransformerModule(d_mode, nhead, num_layers)
        self.transformer.apply(init_weights)

        self.to_vocab = nn.Sequential(nn.Linear(d_mode, len(vocab)))

        self.ce_criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=vocab.word2idx['<pad>'])
        self.nlll_criterion = nn.NLLLoss(reduction="none", ignore_index=vocab.word2idx['<pad>'])

    def forward(self, waveforms, targets):
        _, (image_features, _) = self.model(waveforms)
        image_features = image_features.transpose(1, 2).transpose(1, 0)  # ( batch, feature, number)
        image_features = self.feature_embedding(image_features)
        tgt_key_padding_mask = targets == 0

        embedded = self.embed(targets).transpose(1, 0)
        out = self.transformer(image_features, embedded, tgt_key_padding_mask)
        vocab_distribution = self.to_vocab(out)
        return vocab_distribution

    def loss(self, out, targets):
        out = F.log_softmax(out, dim=-1).reshape(-1, len(self.vocab))
        target = targets[:, 1:]
        batch_size, seq_length = target.shape
        target = target.transpose(1, 0).reshape(-1)
        loss = self.nlll_criterion(out, target)
        loss = loss.reshape(batch_size, seq_length).sum(dim=1).mean(dim=0)

        return loss

    def on_epoch_start(self):
        pass

    def training_step(self, batch, batch_idx):
        waveforms, _, targets, lengths = batch
        vocab_distribution = self.forward(waveforms, targets)[:-1, :, :]
        loss = self.loss(vocab_distribution, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms, _, targets, lengths = batch
        vocab_distribution = self.forward(waveforms, targets)[:-1, :, :]
        loss = self.loss(vocab_distribution, targets)
        return loss

    def on_test_epoch_start(self):
        self.res = {}

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def on_train_end(self):
        pass


class AveragePool(nn.Module):
    def __init__(self, kernel_size=10):
        super(AveragePool, self).__init__()

    def forward(self, x):
        signal_size = x.shape[-1]
        kernel = torch.nn.AvgPool1d(signal_size)
        average_feature = kernel(x).squeeze(-1)
        return x, average_feature


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
