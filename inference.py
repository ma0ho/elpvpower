from argparse import ArgumentParser
import random
from pathlib import Path

import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning import Trainer
import torch as t
from skimage import io, transform

from pvpower.model import PVModel
from pvpower.data import PVPowerCustomDataset
from pvpower.visualization import visualize_fmap


def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', type=Path, default=Path(__file__).parent / 'lightning_logs',
                       help='Path to checkpoint file')
    parser.add_argument('--data_path', type=Path, help='Data folder')
    parser.add_argument('--target_path', type=Path, help='Where to write results')
    parser.add_argument('--num_imgs', type=int, default=-1, help='How many images to process? Default: -1 (=all)')
    return parser


def run_fn(params):
    
    # load model from checkpoint
    model = PVModel.load_from_checkpoint(str(params.checkpoint_path.absolute()), strict=False)
    
    # load data
    data = PVPowerCustomDataset(params.data_path.absolute(), model.test_time_transforms(), params.num_imgs)
    
    # prepare model
    model.eval()
    model = model.cuda()
    
    # accumulate results here
    results = list()
    
    # no gradient needed
    with t.no_grad():
        
        for i in range(len(data)):
            x, _, _, p = data[i]
            x = x.unsqueeze(0).cuda()   # convert to NCHW cuda tensor
            
            if 'model_variant' in model.hparams and model.hparams.model_variant == 'physical':
                # pass through network and perform visualization
                fmap_calibration = np.load(params.checkpoint_path.parent / 'fmap_calibration.npy')
                y, fmap = model(x, return_features=True)
                img = transform.resize(io.imread(p, as_gray=True), (1500, 900))
                calibrated_fmap = fmap.cpu().squeeze().numpy()-fmap_calibration  # subtract constant bias
                visualize_fmap(img.T, calibrated_fmap, params.target_path, p.stem)
            else:
                # pass through network without visualization
                y = model(x)
            
            results.append(dict(path=str(p), relative_power=y.cpu().squeeze().numpy()))
            
    # save results 
    pd.DataFrame(results).to_csv(params.target_path / 'results.csv')
    

if __name__ == '__main__':
    parser = create_parser()
    params = parser.parse_args()
    run_fn(params)