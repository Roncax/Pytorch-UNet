import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def check_create_dir(dir_in):
    if not os.path.exists(dir_in):
        os.mkdir(dir_in)


# return a numpy volume over 1 directory of images (1 patient) - IMAGE MASK
def build_np_volume(dir):
    #TODO parametric shape
    volume = np.empty(shape=(512, 512))
    in_files = os.listdir(dir)
    in_files.sort() # if not sorted we cannot associate gt and prediction

    for i, fn in enumerate(in_files):
        img = Image.open(os.path.join(dir, fn))
        img = np.array(img).astype(np.bool)
        volume = np.dstack((volume, img))

    return volume

