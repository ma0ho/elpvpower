import logging
from enum import Enum
from pathlib import Path

from torch.utils.data import Dataset
import torch as t
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from skimage import io

class PVPowerDataset(Dataset):
    
    class Mode(Enum):
        TEST = 1
        TRAINVAL = 2
        STRATIFIED_TRAIN = 3
        STRATIFIED_VAL = 4

    def __init__(self, mode, data_path, transform, train_subset_query: str = None, seed=None, train_part=0.8, fold_id=None):
        super().__init__()
        self._transform = transform
        self._data_path = data_path
        self._mode = mode
        self._logger = logging.getLogger()
        
        # read data
        tab = pd.read_csv(self._data_path / 'data.csv')
        tab = tab.astype({'nominal_power': float})
        
        # compute relative power for all
        tab['relative_power'] = tab['peak_power']/tab['nominal_power']
        
        # hande CV -> drop not using samples according to fold_id and mode
        if fold_id is not None:
            assert fold_id < 5
            if mode != self.Mode.TEST:
                tab = tab.query('fold_{:d}_train == True'.format(fold_id))
            else:
                tab = tab.query('fold_{:d}_train == False'.format(fold_id))
       
        # this is the data we use in this dataset
        self._tab = tab
       
        # if no specific query is specified for training data
        if train_subset_query is None:
            
            # for stratified train/val -> perform split here
            if mode in (self.Mode.STRATIFIED_TRAIN, self.Mode.STRATIFIED_VAL) and fold_id is not None:
                assert seed is not None    # we need a seed for predictable results

                class_labels = tab['power_group'].values
                train_tab, val_tab = train_test_split(self._tab, stratify=class_labels, train_size=train_part, test_size=1-train_part, random_state=seed)
                self._tab = train_tab if mode == self.Mode.STRATIFIED_TRAIN else val_tab

        else:  # if train set is specified by a pandas query
            assert fold_id is None   # not supported in a CV setting
            
            if mode == self.Mode.TEST:
                self._tab = self._tab.query('not ({})'.format(train_subset_query))   # all data that does *not* satisfy the query
            
            # perform a stratified split on data specified by train_subset_query
            elif mode in (self.Mode.STRATIFIED_TRAIN, self.Mode.STRATIFIED_VAL):
                subtab = tab.query(train_subset_query)
                class_labels = subtab['power_group'].values
                train_tab, val_tab = train_test_split(subtab, stratify=class_labels, train_size=train_part, test_size=1-train_part, random_state=seed)
                self._tab = train_tab if mode == self.Mode.STRATIFIED_TRAIN else val_tab
            elif mode == self.Mode.TRAINVAL:
                self._tab = self._tab.query(train_subset_query)
            else:
                raise RuntimeError('Invalid mode')
            
        self._logger.info('Using {:d} samples'.format(len(self._tab)))
                

    def __getitem__(self, i):
        y_row = self._tab.iloc[i]
        y = y_row['relative_power']

        x = io.imread(self._data_path / 'data' / '{:03d}.png'.format(y_row.name), as_gray=False)
        if self._transform:
            x = self._transform(x)

        return x, t.tensor(y).float(), y_row['module_type'], y_row.name

    def __len__(self):
        return len(self._tab)
    

class PVPowerCustomDataset(Dataset):
    """ Custom Dataset used for inference only """
    
    def __init__(self, data_path: Path, transform, num_imgs: int):
        super().__init__()
        self._transform = transform
        self._data_path = data_path
        self._logger = logging.getLogger()
        
        # find images
        self._imgs = list(data_path.glob('*.png')) + list(data_path.glob('*.tif'))
        if len(self._imgs) > num_imgs and num_imgs != -1:
            self._imgs = self._imgs[:num_imgs]

    def __getitem__(self, i):
        x = io.imread(self._imgs[i], as_gray=False)
        if self._transform:
            x = self._transform(x)

        return x, None, None, self._imgs[i]

    def __len__(self):
        return len(self._imgs)