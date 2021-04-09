import json
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from nibabel import load as load_nii
import h5py

# read and decode a .nii file
def read_nii(path: str):
    img = load_nii(path).get_fdata()
    img = np.uint8(img)
    return img


# load all labels in path from a  .nii
def nii2label(nii_root, root_path, dataset_parameters, paths):
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
def nii2img(nii_root, root_path, dataset_parameters, paths):
    names = [name for name in os.listdir(nii_root)]

    os.makedirs(root_path, exist_ok=True)

    for name in names:
        nii_path = os.path.join(nii_root, name)

        target_path = root_path
        if name in dataset_parameters['test']:
            target_path = f'{paths.dir_test_img}/patient_{name}'
            os.makedirs(target_path, exist_ok=True)

        image_array = read_nii(os.path.join(nii_path, "data.nii"))
        for n in range(image_array.shape[2]):
            cv2.imwrite(os.path.join(target_path, f"patient_{name}_" + 'img{:0>3d}.png'.format(n + 1)),
                        image_array[:, :, n])


# dims: (512,512,n_slice)
def nii_to_hdf5_img(nii_root: str, f: h5py.File, db_info):
    names = [name for name in os.listdir(nii_root)]

    for name in names:
        image_array = read_nii(os.path.join(nii_root, name, "data.nii"))
        mask_array = read_nii(os.path.join(nii_root, name, "label.nii"))

        # save every img slice in a subdir
        for n in range(image_array.shape[2]):
            mode = "test" if name in db_info['test'] else "train"
            f.create_dataset(name=f'{db_info["name"]}/{mode}/volume_{name}/image/slice_{n}', data=image_array[:, :, n])
            f.create_dataset(name=f'{db_info["name"]}/{mode}/volume_{name}/mask/slice_{n}', data=mask_array[:, :, n])


# choose random n volumes and save names in json
def random_split_test(dir: str, db_info: dict, paths):
    names = [name for name in os.listdir(dir)]
    # choose random test images and populate the json file
    test_img = random.sample(names, db_info['numTest'])
    db_info['test'] = test_img
    db_info['train'] = [item for item in names if not item in test_img]

    assert len(db_info['train']) == db_info['numTraining'], f'Invalid number of train items'

    json.dump(db_info, open(paths.json_file, "w"))


# final shape single dataset -> (512,512) mask and img
def prepare_structseg(paths):
    db_info = json.load(open(paths.json_file))

    random_split_test(dir=paths.dir_raw_db, db_info=db_info, paths=paths)
    with h5py.File(f'{paths.dir_database}/{paths.db_name}.hdf5', 'w') as f:
        nii_to_hdf5_img(f=f, db_info=db_info, nii_root=paths.dir_raw_db)

    # with h5py.File(f'{paths.dir_database}/{paths.db_name}.hdf5', 'r') as f:
    #     print(f[f'{db_info["name"]}'].keys())
    #     print(f[f'{db_info["name"]}/train'].keys())
    #     print(f[f'{db_info["name"]}/train/volume_1/mask'].keys())
    #     print(f[f'{db_info["name"]}/train/volume_1/mask/slice_0'].shape)
    #     print(f[f'{db_info["name"]}/train/volume_1/'].keys())
    #     plt.imshow(f[f'{db_info["name"]}/train/volume_1/mask/slice_0'])
    #     plt.show()
    #     print(np.unique(f[f'{db_info["name"]}/train/volume_1/mask/slice_0'][:,:]))
