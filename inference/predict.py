import json
import logging
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utilities.data_vis import plot_img_and_mask, img2gif
import paths
from utilities.various import check_create_dir, build_np_volume
from preprocessing.scale import scale_img
from evaluation.metrics import ConfusionMatrix
import evaluation.metrics as metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from matplotlib import cm


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
            probs = F.softmax(output, dim=1)  # prob from 0 to 1 (dim = masks)
        else:
            probs = torch.sigmoid(output)
        probs = probs.squeeze(0)

        # tf = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize(full_img.size[1]),
        #         transforms.ToTensor()
        #     ]
        # )

        # probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def mask_to_image(mask):
    cm = {
        0: [75, 54, 95],
        1: [153, 255, 102],
        2: [0, 128, 255],
        3: [255, 0, 0],
        4: [172, 88, 193],
        5: [216, 228, 111],
        6: [255, 255, 255]
    }

    finalmask3D = np.empty(shape=(512, 512, 3))
    finalmask_r = np.empty(shape=(512, 512))
    finalmask_g = np.empty(shape=(512, 512))
    finalmask_b = np.empty(shape=(512, 512))

    with open(paths.json_file) as f:
        mask_dict = json.load(f)["labels"]

    for i in range(len(mask_dict)):
        finalmask_r[mask[i]] = cm[i][0]
        finalmask_g[mask[i]] = cm[i][1]
        finalmask_b[mask[i]] = cm[i][2]

    finalmask3D[:, :, 0] = finalmask_r
    finalmask3D[:, :, 1] = finalmask_g
    finalmask3D[:, :, 2] = finalmask_b

    plt.imshow(Image.fromarray(finalmask3D.astype(np.uint8)))
    plt.show()

    return Image.fromarray(finalmask3D.astype(np.uint8))


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

    with open(paths.json_file) as f:
        mask_dict = json.load(f)["labels"]

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
            result = mask_to_image(mask)
            result.save(out_files[i])

        if viz:
            mask = np.array(mask).astype(np.bool)
            plot_img_and_mask(img, mask_to_image(mask), ground_truth=gt_mask, fig_name=fn, patient_name=patient)
    if viz:
        img2gif(png_dir=f"{paths.dir_plot_saves}/{patient}",
                target_folder=paths.dir_predicted_gifs,
                out_name=f"{patient}")

    # build np volume and confusion matrix
    patient_volume = build_np_volume(dir=os.path.join(paths.dir_mask_prediction, patient))
    gt_volume = build_np_volume(dir=os.path.join(paths.dir_test_GTimg, patient))
    for key in mask_dict:
        patient_volume_cp = np.zeros(shape=patient_volume.shape)
        patient_volume_cp[patient_volume == float(key)] = 1
        gt_volume_cp = np.zeros(shape=gt_volume.shape)
        gt_volume_cp[gt_volume == float(key)] = 1

        cm = ConfusionMatrix(test=patient_volume_cp, reference=gt_volume_cp)

        logging.info(
            f"Inference time for {len(in_files)} ({patient}) images: {round(time.time() - t0, 4)} - mean time: {round((time.time() - t0) / len(in_files), 4)}")
        logging.info(f"Metrics for {mask_dict[key]}")

        logging.info(f'Dice: {metrics.ALL_METRICS["Dice"](confusion_matrix=cm)}')
        # for m in metrics.ALL_METRICS.keys():
        #     logging.info(f'{m}: {round(metrics.ALL_METRICS[m](confusion_matrix=cm), 3)}')
