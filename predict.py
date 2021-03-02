import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from dice_loss import dice_coef_test
from structseg_load import img2gif


def predict_img(net,
                full_img,
                device,
                scale_factor,
                out_threshold):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def get_output_filenames_code(path, out_path):
    out_files = []
    input_files = os.listdir(path)

    for f in input_files:
        pathsplit = os.path.splitext(f)
        out_files.append(out_path + "{}_OUT{}".format(pathsplit[0], pathsplit[1]))

    return out_files


def main_code():
    path_in_files = "/home/roncax/Git/Pytorch-UNet/data/Task3_Thoracic_OAR/tests/img/"
    path_out_files = "/home/roncax/Git/Pytorch-UNet/data/Task3_Thoracic_OAR/tests/mask_prediction/"
    path_gt_masks = "/home/roncax/Git/Pytorch-UNet/data/Task3_Thoracic_OAR/tests/img_gt/"

    in_files = os.listdir(path_in_files)
    out_files = get_output_filenames_code(path_in_files, path_out_files)

    model = "/home/roncax/Git/Pytorch-UNet/checkpoints/CP_epoch1.pth"
    scale = 1
    mask_threshold = 0.4
    no_save = False
    viz = True
    dice_scores = []

    net = UNet(n_channels=1, n_classes=1)

    logging.info("Loading model {}".format(model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))

    logging.info("Model loaded!")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(path_in_files + fn))

        img = Image.open(path_in_files + fn)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=scale,
                           out_threshold=mask_threshold,
                           device=device)
        gt_mask = Image.open(path_gt_masks + fn)
        dice = dice_coef_test(gt_mask, mask)

        #TODO trovare soluzione a questo problema
        # to eliminate the problem of 0 intersection value when all background in the dice calc
        if dice != 0:
            dice_scores.append(dice)


        if not no_save:
            result = mask_to_image(mask)
            result.save(out_files[i])
            # logging.info("Mask saved to {}".format(out_files[i]))

        if viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask, ground_truth=gt_mask, dice=dice, fig_name=fn)

    logging.info(f"Mean dice score: {np.array(dice_scores).mean()}")


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)
    main_code()
    img2gif(png_dir="/home/roncax/Git/Pytorch-UNet/data/Task3_Thoracic_OAR/tests/plt_save",
            target_folder="/home/roncax/Git/Pytorch-UNet/data/Task3_Thoracic_OAR/tests/",
            out_name="plots")