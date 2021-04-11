import json
import os

import cv2
import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import logging
from mod_unet.preprocessing.prepare_img import prepare_img, prepare_mask
from mod_unet.preprocessing.ct_levels_enhance import setDicomWinWidthWinCenter


class HDF5Dataset(Dataset):
    def __init__(self, scale: float, db_info: dict, mode: str, paths, labels: dict):
        db_info["experiments"] += 1
        json.dump(db_info, open(paths.json_file, "w"))

        self.labels = labels
        self.db_dir = paths.hdf5_db
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.db_info = db_info
        self.db_info["experiments"] += 1
        self.ids_img = []
        self.ids_mask = []
        db = h5py.File(self.db_dir, 'r')
        # upload data from the hdf5 sctructure
        for volumes in db[f'{self.db_info["name"]}/{mode}'].keys():
            for slice in db[f'{self.db_info["name"]}/{mode}/{volumes}/image'].keys():
                self.ids_img.append(f'{self.db_info["name"]}/{mode}/{volumes}/image/{slice}')
                self.ids_mask.append(f'{self.db_info["name"]}/{mode}/{volumes}/mask/{slice}')
        assert len(self.ids_img) == len(
            self.ids_mask), f"Error in the number of mask {len(self.ids_mask)} and images{len(self.ids_img)}"

        logging.info(f'Creating {mode} dataset with {len(self.ids_img)} images')

    def __len__(self):
        return len(self.ids_img)

    def __getitem__(self, idx):
        db = h5py.File(self.db_dir, 'r')

        img = db[self.ids_img[idx]][()]
        mask = db[self.ids_mask[idx]][()]

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        # only specified classes are considered in the DB
        mask_cp = np.zeros(shape=mask.shape, dtype=int)

        l = [x for x in self.labels.keys() if x != str(0)]

        if len(l) == 1:
            mask_cp[mask == int(l[0])] = 1
            img = setDicomWinWidthWinCenter(img_data=img, winwidth=self.db_info["CTwindow_width"][self.labels[l[0]]],
                                            wincenter=self.db_info["CTwindow_level"][self.labels[l[0]]])
            img = np.uint8(img)

        elif len(l) > 2:
            img = setDicomWinWidthWinCenter(img_data=img, winwidth=self.db_info["CTwindow_width"]["coarse"],
                                            wincenter=self.db_info["CTwindow_level"]["coarse"])
            img = np.uint8(img)

            for key in l:
                mask_cp[mask == int(key)] = key

        img = prepare_img(img, self.scale)
        mask_cp = prepare_mask(mask_cp, self.scale)

        # print(f"MASK: {np.unique(mask)} - {mask.shape} MASK CP: {np.unique(mask_cp)}- {mask_cp.shape} IMG: {np.unique(img)}")
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask_cp).type(torch.FloatTensor),
            'id': self.ids_img[idx]
        }
