from argparse import ArgumentParser
import random
from pathlib import Path

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning import Trainer

from pvpower.model import PVModel


def create_parser():
    parser = ArgumentParser()
    
    ## Basic setup / system related stuff #######################################################################################
    parser.add_argument('--jobname', type=str, default='default',
                       help='Jobname that is used to construct the logging directory')
    parser.add_argument('--ncpus', type=int, default=4,
                       help='Number of CPUs used for data processing')
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Perform a fast development run with only a single batch in every step \
                       of the training schedule')
    parser.add_argument('--log_dir', type=Path, default=Path(__file__).parent / 'lightning_logs',
                       help='This is where logs are stored')
    parser.add_argument('--real_batch_size', default=8, type=int,
                       help='Actual size of batches that need to fit on GPU')
    
    
    ## Set up model #############################################################################################################
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=[
                            'resnet18',
                            'resnet34',
                            'resnet50',
                            'resnet152',
                            'mobilenet_v2'
                        ],
                        help='Choose the type of model')
    parser.add_argument('--model_variant', type=str, choices=['standard', 'physical'], default='standard',
                       help='Specify whether to train the standard model or the physical model \
                       is used. The latter is used to compute the errors per cell.')
    parser.add_argument('--virtual_batch_size', default=8, type=int,
                       help='Real batches are accumulated  such that the logical \
                       batch size corresponds to this')
    parser.add_argument('--optimizer', choices=['Adam', 'SGD', 'AdamW'], default='SGD',
                       help='Choose the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.027546,
                       help='Weight decay using L2-penalty on the weights')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum parameter used in optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.005265,
                       help='Set the learning rate for the optimizer')
    parser.add_argument('--stop_patience', default=40, type=int,
                       help='Training is stopped after this number of epochs without\
                       improvements on the validation set and learning rate is decreased after\
                       stop_patience//2 epochs without improvement')
    parser.add_argument('--normalize', choices=[
                            'imagewise_standardization',
                            'global_standardization',
                            'imagewise_zca_whitening'
                        ],
                        default='global_standardization',
                        help='Choose normalization scheme for images')
    
    ## Data setup ###############################################################################################################
    parser.add_argument('--cv_fold_id', choices=[None,0,1,2,3,4], type=int, default=0,
                       help='Fold of crossvalidation (set None to disable CV)')
    parser.add_argument('--train_subset_query', type=str, default=None,
                       help='Pandas query used to query for the training set. This defaults to \
                       None and is incompatible to the CV. Please disable CV (--cv_fold_id=None) to \
                       make use of this')
    parser.add_argument('--train_percent', type=float, default=1.0,
                       help='Train only on a subset of the training data (1.0=use all data, \
                       0.1=use only 10% and so on). This affects the training as well as validation \
                       set size. Please refer to --train_part to specifiy the fraction of train/validation \
                       split')

    parser.add_argument('--train_part', type=float, default=0.8,
                       help='Fraction of training samples in train/validation split')
    parser.add_argument('--img_size', type=int, default=800,
                       help='Resize smallest edge of images to this size (in px)')

    return parser


def train_fn(hparams):
    ## Set up Lightning Module ##################################################################################################
    data_path = (Path(__file__).parent / 'data').resolve()
    model = PVModel(hparams, data_path)

    ## Set up early stopping ####################################################################################################
    stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=hparams.stop_patience,
        verbose=True,
        mode='min'
    )
    
    ## Set up logging and checkpoints ###########################################################################################
    logger = tt_logger = TestTubeLogger(
        save_dir=hparams.log_dir,
        name=hparams.jobname,
        debug=False,
        create_git_tag=False
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=False,
        monitor='val_loss',
        mode='min',
    )
    
    ## Automatically accumulate batches over multiple GPU passes ################################################################
    acc = hparams.virtual_batch_size // hparams.real_batch_size
    acc = 1 if acc < 1 else acc
    
    # virtual_batch_size must be divisible by real_batch_size
    assert hparams.virtual_batch_size % hparams.real_batch_size == 0
    
    ## Set up Trainer object ####################################################################################################
    trainer = Trainer(
        gpus=1,
        early_stop_callback=stop,
        logger = logger,
        accumulate_grad_batches=acc,
        checkpoint_callback=checkpoint_callback,
        train_percent_check=hparams.train_percent,
        fast_dev_run=hparams.fast_dev_run,
        progress_bar_refresh_rate=10,
    )
    
    ## Start train and testing ##################################################################################################
    trainer.fit(model)
    trainer.test()

if __name__ == '__main__':
    parser = create_parser()
    hparams = parser.parse_args()
    train_fn(hparams)
