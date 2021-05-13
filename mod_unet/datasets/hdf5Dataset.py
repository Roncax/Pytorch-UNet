import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from mod_unet.preprocessing.prepare_img import prepare_img
from mod_unet.preprocessing.ct_levels_enhance import setDicomWinWidthWinCenter
import albumentations as A
from mod_unet.utilities.data_vis import visualize


class HDF5Dataset(Dataset):
    def __init__(self, scale: float, db_info: dict, mode: str, paths, labels: dict, augmentation=False):
        self.db_info = db_info

        self.labels = labels
        self.db_dir = paths.hdf5_db
        self.scale = scale
        self.mode = mode
        self.augmentation = augmentation
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

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
        img_dict = {}

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        if self.mode == "train":

            # only specified classes are considered in the DB
            l = [x for x in self.labels.keys() if x != str(0)]
            mask_cp = np.zeros(shape=mask.shape, dtype=int)

            if len(l) == 1:
                img = setDicomWinWidthWinCenter(img_data=img,
                                                winwidth=self.db_info["CTwindow_width"][self.labels[l[0]]],
                                                wincenter=self.db_info["CTwindow_level"][self.labels[l[0]]])
                mask_cp[mask == int(l[0])] = 1

            else:
                img = setDicomWinWidthWinCenter(img_data=img, winwidth=self.db_info["CTwindow_width"]["coarse"],
                                                wincenter=self.db_info["CTwindow_level"]["coarse"])
                for key in l:
                    mask_cp[mask == int(key)] = key

            img = np.uint8(img)
            img, mask = self.prepare_segmentation_img_mask(img=img, mask=mask_cp)

        # binary mask + multiclass mask and img
        elif self.mode == "test":
            for lab in self.labels:
                img_temp = img.copy()
                img_temp = setDicomWinWidthWinCenter(img_data=img_temp,
                                                     winwidth=self.db_info["CTwindow_width"][self.labels[lab]],
                                                     wincenter=self.db_info["CTwindow_level"][self.labels[lab]])
                img_temp = np.uint8(img_temp)

                # img_temp = prepare_img(img_temp, self.scale)
                img_temp = self.prepare_segmentation_inference_single(img=img_temp)

                img_temp = torch.from_numpy(img_temp).type(torch.FloatTensor)
                img_dict[lab] = img_temp

            l = [x for x in self.labels.keys() if x != str(0)]
            mask_cp = np.zeros(shape=mask.shape, dtype=int)
            img = setDicomWinWidthWinCenter(img_data=img, winwidth=self.db_info["CTwindow_width"]["coarse"],
                                            wincenter=self.db_info["CTwindow_level"]["coarse"])
            for key in l:
                mask_cp[mask == int(key)] = key

            img = np.uint8(img)
            img, mask = self.prepare_segmentation_inference(img=img, mask=mask_cp)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'id': self.ids_img[idx],
            'image_organ': img_dict
        }

    def prepare_segmentation_img_mask(self, img, mask):
        w, h = np.shape(img)
        img = np.expand_dims(img, axis=2)
        mask = np.expand_dims(mask, axis=2)

        resize = A.Resize(height=int(self.scale * w), width=int(self.scale * h), always_apply=True)
        resized_img = resize(image=img, mask=mask)
        original_img = resized_img['image']
        original_mask = resized_img['mask']

        if self.augmentation:
            transform = A.Compose([
                A.ElasticTransform(p=0.5, alpha=120 * 0.25, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=0.5),
                A.RandomScale(scale_limit=0.1, p=0.5),
                A.Rotate(limit=10, p=0.5),
                # A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=10, p=0.5),
                # A.Blur(blur_limit=7, always_apply=False, p=0.5),
                A.GaussNoise(var_limit=(0, 10), always_apply=False, p=0.5),
            ])

            transformed = transform(image=original_img, mask=original_mask)
            img = transformed['image']
            mask = transformed['mask']

            img = img / 255
            # visualize(mask=mask, image=img, original_image=original_img, original_mask=original_mask)

        else:
            img = original_img / 255
            mask = original_mask
            # visualize(mask=mask, image=img)

        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        return img, mask

    def prepare_segmentation_inference(self, img, mask):
        w, h = np.shape(img)
        img = np.expand_dims(img, axis=2)
        mask = np.expand_dims(mask, axis=2)

        resize = A.Resize(height=int(self.scale * w), width=int(self.scale * h), always_apply=True)
        resized_img = resize(image=img, mask=mask)
        original_img = resized_img['image']
        original_mask = resized_img['mask']

        img = original_img / 255
        mask = original_mask
        # visualize(mask=mask, image=img)

        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        return img, mask

    def prepare_segmentation_inference_single(self, img):
        w, h = np.shape(img)
        img = np.expand_dims(img, axis=2)

        resize = A.Resize(height=int(self.scale * w), width=int(self.scale * h), always_apply=True)
        resized_img = resize(image=img)
        original_img = resized_img['image']

        img = original_img / 255
        img = img.transpose((2, 0, 1))

        return img
