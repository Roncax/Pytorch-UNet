import numpy as np

#scale, add channel, normalize and traspose
def prepare_img(img_nd, scale):

    # w, h = pil_img.size
    # newW, newH = int(scale * w), int(scale * h)
    # assert newW > 0 and newH > 0, 'Scale is too small'
    # pil_img = pil_img.resize((newW, newH))
    #
    # img_nd = np.array(pil_img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    img_trans = img_trans / 255

    return img_trans.astype(float)

def prepare_mask(img_nd, scale):
    # w, h = pil_img.size
    # newW, newH = int(scale * w), int(scale * h)
    # assert newW > 0 and newH > 0, 'Scale is too small'
    # pil_img = pil_img.resize((newW, newH))
    #
    # img_nd = np.array(pil_img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)
    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    return img_trans