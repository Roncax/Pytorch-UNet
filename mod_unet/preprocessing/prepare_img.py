import numpy as np
from skimage.transform import resize


# scale, add channel, normalize and traspose
def prepare_img(img_nd, scale):
    if len(img_nd.shape) == 2:
        w, h = img_nd.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        #img_nd = resize(img_nd, (newW, newH))
        img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))

    # normalize
    img_trans = img_trans / 255
    return img_trans


def prepare_mask(img_nd, scale):
    if len(img_nd.shape) == 2:
        w, h = img_nd.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        #img_nd = resize(img_nd, (newW, newH))
        img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    return img_trans
