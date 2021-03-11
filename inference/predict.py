import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from network_architecture import UNet
from utilities import plot_img_and_mask
from dataset_conversion.dataset import BasicDataset
from evaluation.dice_loss import dice_coef_test
from dataset_conversion.structseg_load import img2gif




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

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for f in input_files:
        pathsplit = os.path.splitext(f)
        out_files.append(out_path + "/{}_OUT{}".format(pathsplit[0], pathsplit[1]))

    return out_files


def predict_patient(scale, mask_threshold, no_save, viz, patient, net, device):
    dice_scores = []
    path_in = os.path.join(path_in_files, patient)
    path_out = os.path.join(path_out_files, patient)
    path_gt = os.path.join(path_gt_masks, patient)

    out_files = get_output_filenames_code(path_in, path_out)
    in_files = os.listdir(path_in)

    t0 = time.time()

    intersection_tot = 0
    im_sum_tot = 0
    for i, fn in enumerate(in_files):
        img = Image.open(os.path.join(path_in, fn))

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=scale,
                           out_threshold=mask_threshold,
                           device=device)
        gt_mask = Image.open(os.path.join(path_gt, fn))
        dice, im_sum, intersection = dice_coef_test(gt_mask, mask)

        im_sum_tot += im_sum
        intersection_tot += intersection
        dice_scores.append(dice)

        if not no_save:
            result = mask_to_image(mask)
            result.save(out_files[i])
            # logging.info("Mask saved to {}".format(out_files[i]))

        if viz:
            plot_img_and_mask(img, mask, ground_truth=gt_mask, dice=dice, fig_name=fn, patient_name=patient)
    if viz:
        img2gif(png_dir=f"/databases/Task3_Thoracic_OAR/tests/plt_save/{patient}",
                target_folder="/home/roncax/Git/Pytorch-UNet/databases/Task3_Thoracic_OAR/tests/gifs/",
                out_name=f"{patient}")

    dice_tot = np.array(dice_scores).mean()
    new_dice = (2 * intersection_tot) / im_sum_tot

    end = time.time()
    logging.info(f"Mean dice score of {patient}:\n"
                 f"\tOLD metric: {dice_tot}\n"
                 f"\tNEW metric: {new_dice}")
    logging.info(f"Inference time for {len(in_files)} images: {end - t0} - mean time: {(end - t0) / len(in_files)}")
    return


def predict_total():
    model = "/home/roncax/Git/Pytorch-UNet/checkpoints/CP_EPOCH1-LR(0.0001)_BS(3)_SCALE(1)_EPOCHS(1).pth"
    scale = 1
    mask_threshold = 0.4
    no_save = False
    viz = False

    net = UNet(n_channels=1, n_classes=1)

    logging.info("Loading model {}".format(model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))
    logging.info("Model loaded!")

    for patient in os.listdir(path_in_files):
        predict_patient(scale=scale, mask_threshold=mask_threshold, no_save=no_save, viz=viz, patient=patient, net=net,
                        device=device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    predict_total()
