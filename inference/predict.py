import logging
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utilities.data_vis import plot_img_and_mask
from dataset_conversion.structseg2019_load import img2gif
import paths
from utilities.various import check_create_dir, build_np_volume
from preprocessing.scale import scale_img
from evaluation.metrics import ConfusionMatrix
import evaluation.metrics as metrics


def predict_img(net,
                full_img,
                device,
                scale_factor,
                out_threshold):
    net.eval()
    img = torch.from_numpy(scale_img(full_img, scale_factor))

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

    check_create_dir(out_path)

    for f in input_files:
        pathsplit = os.path.splitext(f)
        out_files.append(out_path + "/{}_OUT{}".format(pathsplit[0], pathsplit[1]))

    return out_files


def predict_patient(scale, mask_threshold, save, viz, patient, net, device):
    path_in = os.path.join(paths.dir_test_img, patient)
    path_out = os.path.join(paths.dir_mask_prediction, patient)
    path_gt = os.path.join(paths.dir_test_GTimg, patient)

    out_files = get_output_filenames_code(path_in, path_out)
    in_files = os.listdir(path_in)

    t0 = time.time()
    for i, fn in enumerate(in_files):
        img = Image.open(os.path.join(path_in, fn))
        gt_mask = Image.open(os.path.join(path_gt, fn))

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=scale,
                           out_threshold=mask_threshold,
                           device=device)



        if save:
            mask = np.array(mask).astype(np.bool)
            print(mask.shape)
            gt_mask = np.array(gt_mask).astype(np.bool)
            print(gt_mask.shape)
            result = mask_to_image(mask)
            result.save(out_files[i])

        if viz:
            plot_img_and_mask(img, mask, ground_truth=gt_mask, fig_name=fn, patient_name=patient)
    if viz:
        img2gif(png_dir=f"{paths.dir_plot_saves}/{patient}",
                target_folder=paths.dir_predicted_gifs,
                out_name=f"{patient}")

    if net.n_classes == 1:
        #build np volume and confusion matrix
        patient_volume = build_np_volume(dir=os.path.join(paths.dir_test_GTimg, patient))
        gt_volume = build_np_volume(dir=os.path.join(paths.dir_mask_prediction, patient))
        cm = ConfusionMatrix(test=patient_volume, reference=gt_volume)

        # print results
        logging.info(
            f"Inference time for {len(in_files)} ({patient}) images: {round(time.time() - t0, 4)} - mean time: {round((time.time() - t0) / len(in_files), 4)}")

        for m in metrics.ALL_METRICS.keys():
            logging.info(f'{m}: {round(metrics.ALL_METRICS[m](confusion_matrix=cm), 3)}')

    return
