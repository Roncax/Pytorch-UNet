import numpy as np
import math
import torchvision


def get_closest_tile_size(desired_size, maximum_size):
    desired_size += 184
    k = int(math.ceil((desired_size - 28) / 16))
    k = max(k, 10)  # k=10 is the smallest possible input

    if k > 10 and k * 16 + 28 - 184 > maximum_size:
        k -= 1

    if k * 16 + 28 - 184 > maximum_size:
        raise Exception(f'Cannot find any tile smaller than {maximum_size}')

    return k * 16 + 28 - 184


def get_overlap_tiles_coords(image_height, image_width, tile_height, tile_width):
    n_tiles_x = int(math.ceil(image_width / tile_width))
    n_tiles_y = int(math.ceil(image_height / tile_height))
    if n_tiles_y > 1 and n_tiles_x > 1:
        step_x = (image_width - tile_width) // (n_tiles_x - 1)
        step_y = (image_height - tile_height) // (n_tiles_y - 1)
        coords = [(j * step_y, i * step_x) for i in range(n_tiles_x) for j in range(n_tiles_y)]
    else:
        coords = [(0, 0)]
    return coords


def get_overlap_tiles(pil_img, tiles_coords, tile_height, tile_width, padding=0):
    if padding > 0:
        pil_img = torchvision.transforms.functional.pad(pil_img, padding=padding, padding_mode='reflect')

    tiles = []
    for tile in tiles_coords:
        img = torchvision.transforms.functional.crop(pil_img, top=tile[0], left=tile[1],
                                                     height=tile_height + 2 * padding, width=tile_width + 2 * padding)
        tiles.append(img)

    return tiles


def stitch_tiles(tiles, tiles_coords):
    channels, tile_h, tile_w = tiles[0].shape
    max_y = max([tile[0] for tile in tiles_coords])
    max_x = max([tile[1] for tile in tiles_coords])
    img_h = max_y + tile_h
    img_w = max_x + tile_w
    image = np.zeros((channels, img_h, img_w))
    count = np.zeros((img_h, img_w))

    for tile, coord in zip(tiles, tiles_coords):
        count[coord[0]:coord[0]+tile_h, coord[1]:coord[1]+tile_w] += 1
        image[:, coord[0]:coord[0]+tile_h, coord[1]:coord[1]+tile_w] += tile

    # average the parts that were on multiple tiles
    image /= count
    return image


if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    import torchvision

    plt.axis('off')
    img = Image.open('../data/imgs/.test.jpg')
    mask = Image.open('../data/masks/.test.gif')
    w, h = img.size
    tile_w = get_closest_tile_size(w / 2, w)
    tile_h = get_closest_tile_size(h / 2, h)
    coords = get_overlap_tiles_coords(h, w, tile_h, tile_w)

    tiles = get_overlap_tiles(mask, coords, tile_h, tile_w, padding=0)
    tiles = list(map(lambda x: torchvision.transforms.functional.pil_to_tensor(x).numpy(), tiles))
    reconstructed = stitch_tiles(tiles, coords)
    assert (np.array(mask) == reconstructed[0, :, :]).all(), \
        "Reconstructed mask does not match initial mask"
    print("Mask OK")

    tiles = get_overlap_tiles(img, coords, tile_h, tile_w, padding=0)
    tiles = list(map(lambda x: torchvision.transforms.functional.pil_to_tensor(x).numpy(), tiles))

    reconstructed = stitch_tiles(tiles, coords).astype(np.int)
    plt.imshow(reconstructed.transpose((1, 2, 0)))
    assert (np.array(img).transpose((2, 0, 1)) == reconstructed).all(), \
        "Reconstructed image does not match initial image"
    print("Image OK")