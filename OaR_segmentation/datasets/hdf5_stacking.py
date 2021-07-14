from numpy.lib.arraysetops import unique
from OaR_segmentation.utilities.data_vis import visualize
import h5py
import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from OaR_segmentation.preprocessing.ct_levels_enhance import setDicomWinWidthWinCenter
from OaR_segmentation.preprocessing.prepare_augment_dataset import *


class HDF5Dataset_stacking(Dataset):
    def __init__(self, scale: float, paths, labels: dict,  channels, augmentation=False,):

        self.labels = labels
        self.db_dir = paths.hdf5_stacking
        self.scale = scale
        self.augmentation = augmentation
        self.channels = channels
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids_mask = []
        db = h5py.File(self.db_dir, 'r')
        # upload data from the hdf5 sctructure
        for n in db.keys():
            self.ids_mask.append(n)


        logging.info(f'Creating stacking dataset with {len(self.ids_mask)} images')

    def __len__(self):
        return len(self.ids_mask)

    def __getitem__(self, idx):
        db = h5py.File(self.db_dir, 'r')
        #print(db["1"].keys())
        #print(f"MASK {self.ids_mask[idx]}")

        masks = db[self.ids_mask[idx]]


        final_array = np.empty(shape=(1,512,512))
        for mask_name in masks.keys():
           
            if mask_name != "gt":

                
                t_mask_dict = prepare_segmentation_mask(mask=masks[mask_name], scale=self.scale)
                visualize(image=t_mask_dict.squeeze(), mask=t_mask_dict.squeeze())

                final_array = np.concatenate((t_mask_dict, final_array), axis=0)
            else:
                temp_mask = prepare_segmentation_mask(mask=masks[mask_name], scale=self.scale)
                #temp_mask = np.uint8(temp_mask)
                gt_mask = temp_mask
                visualize(image=gt_mask.squeeze(), mask=gt_mask.squeeze())

        


        return {
                    'image': torch.from_numpy(final_array).type(torch.FloatTensor),
                    'mask': torch.from_numpy(gt_mask).type(torch.FloatTensor),
                    'id': self.ids_mask[idx]
                }