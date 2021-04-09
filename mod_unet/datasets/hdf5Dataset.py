import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from mod_unet.preprocessing.prepare_img import prepare_img, prepare_mask


class HDF5Dataset(Dataset):
    def __init__(self, scale: float, db: h5py.File, db_info: dict, mode: str, binary_label: str):
        self.db = db
        self.scale = scale
        self.binary_label=binary_label
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.db_info = db_info
        self.db_info["experiments"] += 1
        self.ids_img = []
        self.ids_mask = []

        # upload data from the hdf5 sctructure
        for volumes in db[f'{db_info["name"]}/{mode}'].keys():
            for slice in db[f'{db_info["name"]}/{mode}/{volumes}/image'].keys():
                self.ids_img.append(f'{db_info["name"]}/{mode}/{volumes}/image/{slice}')
                self.ids_mask.append(f'{db_info["name"]}/{mode}/{volumes}/mask/{slice}')
        assert len(self.ids_img) == len(
            self.ids_mask), f"Error in the number of mask {len(self.ids_mask)} and images{len(self.ids_img)}"

        logging.info(f'Creating dataset with {len(self.ids_img)} images')

    def __len__(self):
        return len(self.ids_img)

    def __getitem__(self, idx):

        img = self.db[self.ids_img[idx]]
        mask = self.db[self.ids_mask[idx]]

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'


        img = prepare_img(img, self.scale)
        mask = prepare_mask(mask, self.scale)



        if self.binary_label is not None:
            mask_cp = np.zeros(shape=mask.shape, dtype=int)
            mask_cp[mask == int(self.binary_label)] = 1
            # print(f"MASK: {np.unique(mask)} MASK CP: {np.unique(mask_cp)}, {self.binary_label}")
            mask = mask_cp

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
