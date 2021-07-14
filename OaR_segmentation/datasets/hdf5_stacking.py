from OaR_segmentation.utilities.data_vis import visualize
import h5py
import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from OaR_segmentation.preprocessing.ct_levels_enhance import setDicomWinWidthWinCenter
from OaR_segmentation.preprocessing.prepare_augment_dataset import *


class HDF5Dataset(Dataset):
    def __init__(self, scale: float, db_info: dict, mode: str, paths, labels: dict,  channels, augmentation=False,):
        self.db_info = db_info

        self.labels = labels
        self.db_dir = paths.hdf5_stacking
        self.scale = scale
        self.mode = mode
        self.augmentation = augmentation
        self.channels = channels
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids_mask = []
        db = h5py.File(self.db_dir, 'r')
        # upload data from the hdf5 sctructure
        for volumes in db[f'{self.db_info["name"]}/{mode}'].keys():
            ks = db[f'{self.db_info["name"]}/{mode}/{volumes}/image'].keys()
            for slice in ks:
                self.ids_mask.append(f'{self.db_info["name"]}/{mode}/{volumes}/mask/{slice}')


        logging.info(f'Creating {mode} dataset with {len(self.ids_img)} images')

    def __len__(self):
        return len(self.ids_img)

    def __getitem__(self, idx):
        db = h5py.File(self.db_dir, 'r')
        mask = db[self.ids_mask[idx]][()]
        
        print(mask)
        # dict with labels and gt        


        return {
                    'pred_masks': ,
                    'gt': ,
                    'id': self.ids_img[idx],
                }