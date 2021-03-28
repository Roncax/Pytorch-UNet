import json
import os
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

import paths
from preprocessing.scale import prepare_img, prepare_mask


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.dataset_parameters = json.load(open(paths.json_file))
        self.dataset_parameters["experiments"] += 1
        json.dump(self.dataset_parameters, open(paths.json_file, "w"))

        self.ids = [name for name in os.listdir(imgs_dir)]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = os.path.join(self.masks_dir, idx)
        img_file = os.path.join(self.imgs_dir, idx)

        mask = Image.open(mask_file)
        img = Image.open(img_file)

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = prepare_img(img, self.scale)
        mask = prepare_mask(mask, self.scale)


        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
