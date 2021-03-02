import os
import cv2
from nibabel import load as load_nii
import numpy as np
import imageio

raw_data_path = "/home/roncax/Git/Pytorch-UNet/data/Task3_Thoracic_OAR/Thoracic_OAR"
data_path = "/home/roncax/Git/Pytorch-UNet/data/Task3_Thoracic_OAR/Thoracic_OAR_img/images"
mask_path = "/home/roncax/Git/Pytorch-UNet/data/Task3_Thoracic_OAR/Thoracic_OAR_img/masks"
test_path = "/home/roncax/Git/Pytorch-UNet/data/Task3_Thoracic_OAR/tests"

patients_test = ['25', '30', '50']


# read and decode a .nii file
def read_nii(path):
    img = load_nii(path).get_fdata()
    img = setDicomWinWidthWinCenter(img, winwidth=1800, wincenter=-500)  # liver 300 -20
    img = np.uint8(img)
    return img


# load all labels in path from a  .nii
def nii2label(nii_root, root_path):
    names = [name for name in os.listdir(nii_root)]
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    for name in names:
        nii_path = os.path.join(nii_root, name)

        target_path = root_path
        if name in patients_test:
            target_path = os.path.join(test_path, f"img_gt/patient_{name}")
            if not os.path.exists(target_path):
                os.mkdir(target_path)

        label_array = np.uint8(load_nii(os.path.join(nii_path, "label.nii")).get_fdata())

        # different colors by organs (label_array are different masks)
        label_array[label_array == 1] = 255  # foreground
        label_array[label_array == 2] = 150  # lungs (both)
        # label_array[label_array == 3] = 150
        # label_array[label_array == 4] = 50
        # label_array[label_array == 5] = 125
        # label_array[label_array == 6] = 200
        # label_array[label_array == 7] = 175

        print(f"{name}: {label_array.shape}")

        # save labels with patient's number
        for n in range(label_array.shape[2]):
            cv2.imwrite(os.path.join(target_path, f"patient_{name}_" + 'img{:0>3d}.png'.format(n + 1)),
                        label_array[:, :, n])


# load all images in path from a  .nii
def nii2img(nii_root, root_path):
    names = [name for name in os.listdir(nii_root)]
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    for name in names:
        nii_path = os.path.join(nii_root, name)

        target_path = root_path
        if name in patients_test:
            target_path = os.path.join(test_path, f"img/patient_{name}")
            if not os.path.exists(target_path):
                os.mkdir(target_path)

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


# create a gif from images of the target folder
def img2gif(png_dir, target_folder, out_name="data"):
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(target_folder + f"{out_name}.gif", images)


def load_all():
    nii2img(nii_root=raw_data_path, root_path=data_path)
    nii2label(nii_root=raw_data_path, root_path=mask_path)


load_all()
