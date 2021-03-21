import numpy as np


def scale_img(pil_img, scale):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    pil_img = pil_img.resize((newW, newH))

    img_nd = np.array(pil_img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)
    else:
        # grayscale input image
         #scale between 0 and 1
         img_nd = img_nd / 255

    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    return img_trans.astype(float)

def scale_mask(pil_img, scale):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    pil_img = pil_img.resize((newW, newH))

    img_nd = np.array(pil_img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)
    #else:
        # grayscale input image
        # scale between 0 and 1
        # img_nd = img_nd / 255

    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    # if img_trans.max() > 1:
    #     img_trans = img_trans / 255

    return img_trans