import json
import os
import random
import cv2
import numpy as np
from nibabel import load as load_nii

import paths
from preprocessing.ct_levels_enhance import setDicomWinWidthWinCenter

dataset_parameters = json.load(open(paths.json_file))


# read and decode a .nii file
def read_nii(path):
    img = load_nii(path).get_fdata()
    img = setDicomWinWidthWinCenter(img, winwidth=dataset_parameters['CTwindow_width']['coarse'],
                                    wincenter=dataset_parameters['CTwindow_level']['coarse'])
    img = np.uint8(img)
    return img


# load all labels in path from a  .nii
def nii2label(nii_root, root_path):
    names = [name for name in os.listdir(nii_root)]
    os.makedirs(root_path, exist_ok=True)

    for name in names:
        nii_path = os.path.join(nii_root, name)

        target_path = root_path
        if name in dataset_parameters['test']:
            target_path = f'{paths.dir_test_GTimg}/patient_{name}'
            os.makedirs(target_path)

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

    os.makedirs(root_path)

    for name in names:
        nii_path = os.path.join(nii_root, name)

        target_path = root_path
        if name in dataset_parameters['test']:
            target_path = f'{paths.dir_test_img}/patient_{name}'
            os.makedirs(target_path)

        image_array = read_nii(os.path.join(nii_path, "data.nii"))
        for n in range(image_array.shape[2]):
            cv2.imwrite(os.path.join(target_path, f"patient_{name}_" + 'img{:0>3d}.png'.format(n + 1)),
                        image_array[:, :, n])


def random_split_test(dir):
    names = [name for name in os.listdir(dir)]
    # choose random test images and populate the json file
    test_img = random.sample(names, dataset_parameters['numTest'])
    dataset_parameters['test'] = test_img
    dataset_parameters['train'] = [item for item in names if not item in test_img]
    assert len(dataset_parameters['train']) == dataset_parameters['numTraining'], f'Invalid number of train items'
    json.dump(dataset_parameters, open(paths.json_file, "w"))


if __name__ == '__main__':
    random_split_test(dir=paths.dir_raw_db)
    nii2img(nii_root=paths.dir_raw_db, root_path=paths.dir_train_imgs)
    nii2label(nii_root=paths.dir_raw_db, root_path=paths.dir_train_masks)
