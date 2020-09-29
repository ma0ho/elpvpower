import os
from copy import deepcopy
from pathlib import Path
import gc
import random

import pandas as pd
from matplotlib import pyplot as plt

import torch as t
from torch.nn import functional as F
from torch.nn import Sequential, Flatten, MSELoss, Linear, ReLU, Dropout, Identity, CrossEntropyLoss, Conv2d, Module, LPPool2d, AdaptiveAvgPool2d
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torchvision.models as models

import pytorch_lightning as pl

from .data import PVPowerDataset
from .transforms import *
from .modules import *

class PVModel(pl.LightningModule):

    def init_model(self):
        self._features = None
        self._regression = None

        # load pretrained model and re-initialize fully connected layer
        factory = getattr(models, self.hparams.model)
        net = factory(pretrained=True)
        if self.hparams.model in ('resnet18', 'resnet34'):
            embedding_size = 512
        elif self.hparams.model == 'mobilenet_v2':
            embedding_size = net.last_channel
        else:
            embedding_size = 2048

        if 'resn' in self.hparams.model:
            net.fc = Identity()
            regressor = Linear(embedding_size, 1)
        elif self.hparams.model == 'mobilenet_v2':
            net.classifier = Identity()
            regressor = Sequential(
                Dropout(0.2),
                Linear(embedding_size, 1),
            )
        
        # default variant
        if 'model_variant' not in self.hparams or self.hparams.model_variant == 'standard':
            self._features = net
            self._regression = Sequential(
                regressor,
                Flatten(0,-1)
            )
            
        # variant used to generate class activation maps
        elif self.hparams.model_variant == 'physical':
            assert self.hparams.model == 'resnet18'
            
            # register a custom forward method that omits
            # fully connected and global average pooling
            def fw(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                return x
            net.forward = fw.__get__(net)
            
            # set up head as described in the paper
            self._features = Sequential(
                net,
                Conv2d(in_channels=512, out_channels=1, kernel_size=(1,1)),
                ReLU(),
                MulConst(-1),
                MulConst(1/(40*40)),
            )
            self._regression = Sequential(
                SumPool(),
                Flatten(0, -1),
                AddOne(),
            )
            
        # backwards compability to older checkpoints
        self._model = self._features
        self._pv_estimator = self._regression

        # randomize train/validation splits
        self._ds_random_seed = random.randint(0,2**32)
        
        # set example input to allow graph construction
        self.example_input_array = t.rand(8, 3, self.hparams.img_size, self.hparams.img_size)

    def __init__(self, hparams, data_path=None):
        super().__init__()

        self.hparams = hparams
        self._data_path = data_path

        self.init_model()
        self._crit = MSELoss()
        self._crit_noreduce = MSELoss(reduction='none')

        # reduce on plateau
        # TODO: Fix this strange LR calculation and update learning rates in paper
        # and update default learning rate in ArgumentParser
        self._max_lr = self.hparams.learning_rate
        self._base_lr = self._max_lr/4
        self._initial_lr = (self._max_lr-self._base_lr)/2 + self._base_lr
        self._patience = self.hparams.stop_patience if 'stop_patience' in self.hparams.keys() else self.hparams.lr_patience

        # normalization
        if hparams.normalize == 'imagewise_standardization':
            self._normalize_transform = PerImageNormalization()
        elif hparams.normalize == 'global_standardization':
            self._normalize_transform = transforms.Normalize([0.13226703, 0.13226703, 0.13226703], [0.05048449, 0.05048449, 0.05048449])
        elif hparams.normalize == 'imagewise_zca_whitening':
            self._normalize_transform = PerImageZCAWhitening()

    def forward(self, x, return_features=False):
        f = self._features.forward(x)
        y = self._regression(f)
        
        if return_features:
            return y, f
        else:
            return y

    def training_step(self, batch, batch_nb):
        x, y, _, _ = batch   # drop source, sample id

        # prediction
        y_hat = self.forward(x)
        
        # loss
        loss = self._crit.forward(y_hat, y)

        # log loss
        tensorboard_logs = dict(
            train_loss=loss
        )

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y, sources, _ = batch   # drop sample id
        
        # compute per sample losses
        y_hat = self.forward(x)
        losses = self._crit_noreduce.forward(y_hat, y)
        
        return {'val_losses': losses, 'sources': sources}

    def validation_epoch_end(self, outputs):
        # compute average loss over validation set
        avg_loss = t.cat([x['val_losses'] for x in outputs], axis=0).mean()
        
        # compute losses by source
        all_losses = np.concatenate([x['val_losses'].cpu().numpy().flatten() for x in outputs])
        all_sources = list()
        for x in outputs:
            all_sources += list(x['sources'])
        losses_by_source = dict()
        for source in set(all_sources):
            sel = [x == source for x in all_sources]
            losses_by_source['val_loss_{}'.format(source)] = np.mean(all_losses[sel]) if np.sum(sel) > 0 else 0.0
            
        # log
        tensorboard_logs = dict(val_loss=avg_loss)
        tensorboard_logs.update(losses_by_source)

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y, sources, sample_ids = batch

        # compute prediction
        y_hat = self.forward(x)

        return {'test_errors': y_hat-y, 'sources': sources, 'sample_ids': sample_ids}

    def test_epoch_end(self, outputs):
        # compute averate test loss
        avg_loss = t.cat([x['test_errors'].reshape(-1).abs() for x in outputs]).mean()
        
        # compute losses by source
        all_errors = np.concatenate([x['test_errors'].cpu().numpy().flatten() for x in outputs])
        all_sources = list()
        for x in outputs:
            all_sources += list(x['sources'])
        all_ids = np.concatenate([x['sample_ids'].cpu().numpy().flatten() for x in outputs])
        losses_by_source = dict()
        for source in set(all_sources):
            sel = [x == source for x in all_sources]
            losses_by_source['test_loss_{}'.format(source)] = np.mean(np.abs(all_errors[sel])) if np.sum(sel) > 0 else 0.0
            
        # log
        tensorboard_logs = {'test_loss': avg_loss}
        tensorboard_logs.update(losses_by_source)

        # save errors per sample
        exp_dir = Path(self.logger.experiment.get_data_path(self.logger.name, self.logger.version))
        df = pd.DataFrame({'error': all_errors}, index=all_ids)
        df.to_csv(exp_dir / 'sample_errors.csv')

        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        if self.hparams.optimizer == 'SGD':
            optim = t.optim.SGD(self.parameters(True), lr=self._initial_lr, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum)
        elif self.hparams.optimizer == 'Adam':
            optim = t.optim.Adam(self.parameters(True), lr=self._initial_lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'AdamW':
            optim = t.optim.AdamW(self.parameters(True), lr=self._initial_lr, weight_decay=self.hparams.weight_decay)

        scheduler = dict(
                # automatically reduce LR if no improvement over 
            scheduler = ReduceLROnPlateau(optim, 'min', patience=self._patience//2, verbose=True),
            interval = 'epoch',
            frequency = 1,
            monitor='val_loss'
        )
            
        return [optim], [scheduler]

    def test_time_transforms(self):
        t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.hparams.img_size),
            transforms.ToTensor(),
            self._normalize_transform
        ])
        return t

    def train_dataloader(self):
        # data augmentation during training
        t = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine((0, 10), translate=(0, 0.05), scale=(0.8, 1.0)),
            transforms.Resize(self.hparams.img_size),
            transforms.ToTensor(),
            self._normalize_transform
        ]
        t = transforms.Compose(t)

        # set up dataset
        mode = PVPowerDataset.Mode.STRATIFIED_TRAIN
        ds = PVPowerDataset(
            mode=PVPowerDataset.Mode.STRATIFIED_TRAIN,
            data_path=self._data_path,
            transform=t,
            train_subset_query=self.hparams.train_subset_query,
            seed=self._ds_random_seed,
            train_part=self.hparams.train_part,
            fold_id=self.hparams.cv_fold_id
        )
        
        # set up data loader
        return DataLoader(ds, shuffle=True, num_workers=self.hparams.ncpus, batch_size=self.hparams.real_batch_size)

    def val_dataloader(self):
        # set up dataset using test time transforms and validation split
        ds = PVPowerDataset(
            mode=PVPowerDataset.Mode.STRATIFIED_VAL,
            data_path=self._data_path,
            transform=self.test_time_transforms(),
            train_subset_query=self.hparams.train_subset_query,
            seed=self._ds_random_seed,
            train_part=self.hparams.train_part,
            fold_id=self.hparams.cv_fold_id
        )
        
        # set up data loader
        return DataLoader(ds, shuffle=False, num_workers=self.hparams.ncpus, batch_size=self.hparams.real_batch_size)

    def test_dataloader(self):
        # set up dataset using test time transforms
        ds = PVPowerDataset(
            mode=PVPowerDataset.Mode.TEST,
            data_path=self._data_path,
            transform=self.test_time_transforms(),
            train_subset_query=self.hparams.train_subset_query,
            fold_id=self.hparams.cv_fold_id
        )
        
        # set up data loader
        return DataLoader(ds, shuffle=False, num_workers=self.hparams.ncpus, batch_size=self.hparams.real_batch_size)
