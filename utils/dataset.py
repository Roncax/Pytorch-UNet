from os.path import splitext, join
from os import listdir
from torch.utils.data import Dataset
import logging
from PIL import Image
import torchvision
from .overlap_tiles import get_closest_tile_size, get_overlap_tiles_coords
import random


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, n_tiles=6):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.n_tiles = n_tiles

        self.images = [file for file in listdir(imgs_dir)
                       if not file.startswith('.')]
        if len(self.imgs_dir) == 0:
            logging.error(f'No file found in directory {self.imgs_dir}')
        else:
            logging.info(f'Creating dataset with {len(self.imgs_dir)} examples')

    def __len__(self):
        return len(self.images)

    def mask_name_from_image_name(self, image_name):
        base = splitext(image_name)[0]
        return base + '_mask.gif'

    def preprocess(self, pil_img, pil_mask):
        padded_img = torchvision.transforms.functional.pad(pil_img, padding=92, padding_mode='reflect')
        w, h = pil_img.size
        tile_w = get_closest_tile_size(w / self.n_tiles, w)
        tile_h = get_closest_tile_size(h / self.n_tiles, h)

        tiles = get_overlap_tiles_coords(h, w, tile_h, tile_w)
        random_tile = random.choice(tiles)

        mask = torchvision.transforms.functional.crop(pil_mask, top=random_tile[0], left=random_tile[1],
                                                      height=tile_h, width=tile_w)
        img = torchvision.transforms.functional.crop(padded_img, top=random_tile[0], left=random_tile[1],
                                                     height=tile_h + 184, width=tile_w + 184)

        mask_tensor = torchvision.transforms.functional.pil_to_tensor(mask).reshape(tile_h, tile_w)
        img_tensor = torchvision.transforms.functional.pil_to_tensor(img)

        # img_tensor = torchvision.transforms.functional.normalize(img_tensor, mean=[127.0]*3, std=[127.0]*3)

        return img_tensor, mask_tensor

    def __getitem__(self, i):
        img_name = self.images[i]
        mask_file = join(self.masks_dir, self.mask_name_from_image_name(img_name))
        img_file = join(self.imgs_dir, img_name)

        mask = Image.open(mask_file)
        img = Image.open(img_file)

        assert img.size == mask.size, \
            f'Image and mask {img_name} should be the same size, but are {img.size} and {mask.size}'

        img, mask = self.preprocess(img, mask)
        return {
            'image': img,
            'mask': mask
        }


if __name__ == '__main__':
    ds = BasicDataset('/mnt/hdd/datasets/train_hq', '/mnt/hdd/datasets/train_masks', 2)
    data = ds[0]
    import matplotlib.pyplot as plt
    plt.imshow(data['image'].numpy().transpose((1, 2, 0)))
    plt.show()
    plt.imshow(data['mask'].numpy().transpose((1, 2, 0)))
    plt.show()
    print(data['image'].shape, data['mask'].shape)