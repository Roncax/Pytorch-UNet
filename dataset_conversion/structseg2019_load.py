import json
import os

import cv2
import numpy as np
from nibabel import load as load_nii

import paths
from utilities.various import check_create_dir


with open(paths.json_file) as f:
    test_list = json.load(f)["test"]

# read and decode a .nii file
def read_nii(path):
    img = load_nii(path).get_fdata()
    img = setDicomWinWidthWinCenter(img, winwidth=1800, wincenter=-500)  # liver 300 -20
    img = np.uint8(img)
    return img


# load all labels in path from a  .nii
def nii2label(nii_root, root_path):
    names = [name for name in os.listdir(nii_root)]
    check_create_dir(root_path)

    for name in names:
        nii_path = os.path.join(nii_root, name)

        target_path = root_path
        if name in test_list:
            target_path = f'{paths.dir_test_GTimg}/patient_{name}'
            check_create_dir(target_path)

        label_array = np.uint8(load_nii(f'{nii_path}/label.nii').get_fdata())

        # organs = ['Bg', 'RightLung', 'LeftLung', 'Heart', 'Trachea', 'Esophagus', 'SpinalCord']
        # label_array[label_array == 0] = 0
        # label_array[label_array == 1] = 1
        # label_array[label_array == 2] = 2
        # label_array[label_array == 3] = 3
        # label_array[label_array == 4] = 4
        # label_array[label_array == 5] = 5
        # label_array[label_array == 6] = 6

        # save labels with patient's number
        for n in range(label_array.shape[2]):
            cv2.imwrite(os.path.join(target_path, f"patient_{name}_" + 'img{:0>3d}.png'.format(n + 1)),
                        label_array[:, :, n])


# load all images in path from a  .nii
def nii2img(nii_root, root_path):
    names = [name for name in os.listdir(nii_root)]
    check_create_dir(root_path)

    for name in names:
        nii_path = os.path.join(nii_root, name)

        target_path = root_path
        if name in test_list:
            target_path = f'{paths.dir_test_img}/patient_{name}'
            check_create_dir(target_path)

        image_array = read_nii(os.path.join(nii_path, "data.nii"))
        print(f"{name}: {image_array.shape}")
        for n in range(image_array.shape[2]):
            cv2.imwrite(os.path.join(target_path, f"patient_{name}_" + 'img{:0>3d}.png'.format(n + 1)),
                        image_array[:, :, n])


def setDicomWinWidthWinCenter(img_data, winwidth, wincenter):
    img_temp = img_data
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    img_temp = (img_temp - min) * dFactor

    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255

    return img_temp


def load_all():
    nii2img(nii_root=paths.dir_raw_db, root_path=paths.dir_train_imgs)
    nii2label(nii_root=paths.dir_raw_db, root_path=paths.dir_train_masks)

