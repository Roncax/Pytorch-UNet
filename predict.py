import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import PIL
import torchvision

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.overlap_tiles import get_closest_tile_size, get_overlap_tiles_coords, get_overlap_tiles, stitch_tiles


def predict_img(net: UNet,
                img: PIL.Image,
                device,
                n_tiles):
    net.eval()

    # 1. Get the tile size based on the number of tiles wanted in each dimension (width and height)
    w, h = img.size
    tile_w = get_closest_tile_size(w / n_tiles, w)
    tile_h = get_closest_tile_size(h / n_tiles, h)

    # 2. Split the image into tiles (with padding for the border tiles)
    tiles_coords = get_overlap_tiles_coords(h, w, tile_h, tile_w)
    tiles = get_overlap_tiles(img, tiles_coords, tile_h, tile_w, padding=92)

    # 3. Get a prediction from the network for each tile
    mask_tiles = []
    with torch.no_grad():
        for tile in tiles:
            img_tensor = torchvision.transforms.functional.pil_to_tensor(tile)\
                .unsqueeze(0).to(device=device, dtype=torch.float32)
            output = net(img_tensor)
            if net.n_classes > 1:
                probs = F.softmax(output, dim=1)
            else:
                probs = torch.sigmoid(output)
            mask_tiles.append(probs.squeeze(0).cpu().numpy())

    # 4. Stitch together the tile predictions into the final mask
    mask = stitch_tiles(mask_tiles, tiles_coords).squeeze()
    return mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--channels', type=int,
                        help='Number of channels of the input images',
                        default=3)
    parser.add_argument('--classes', type=int,
                        help='Number of classes of the output mask (including background)',
                        default=2)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = args.output

    if not out_files:
        out_files = [f'{base}_mask{ext}' for [base, ext] in map(os.path.splitext, in_files)]
    elif len(in_files) != len(out_files):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()

    return out_files


def mask_to_image(mask):
    return Image.fromarray(np.uint8((mask * 255)))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=args.channels, n_classes=args.classes)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    # net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        mask = predict_img(net=net,
                           img=img,
                           n_tiles=2,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
