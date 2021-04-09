import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from utilities.data_vis import save_img_mask_plot, img2gif
from preprocessing.prepare_img import prepare_img
from evaluation.eval import eval_inference
from utilities.build_volume import mask_to_image1D, mask_to_image3D


def predict_img(net,
                full_img,
                device,
                scale_factor,
                out_threshold, labels):
    net.eval()
    img = torch.from_numpy(prepare_img(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)  # prob from 0 to 1 (dim = masks)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_output_filenames_code(path, out_path):
    out_files = []
    input_files = os.listdir(path)
    os.makedirs(out_path, exist_ok=True)

    for f in input_files:
        pathsplit = os.path.splitext(f)
        out_files.append(out_path + "/{}_OUT{}".format(pathsplit[0], pathsplit[1]))

    return out_files


def predict_patient(scale, mask_threshold, viz, patient, net, device, paths, labels):
    path_in = os.path.join(paths.dir_test_img, patient)
    path_out = os.path.join(paths.dir_mask_prediction, patient)
    path_gt = os.path.join(paths.dir_test_GTimg, patient)

    out_files = get_output_filenames_code(path_in, path_out)
    in_files = os.listdir(path_in)

    with open(paths.json_file) as f:
        mask_dict = json.load(f)["labels"]

    for i, fn in enumerate(in_files):
        img = Image.open(os.path.join(path_in, fn))
        gt_mask = Image.open(os.path.join(path_gt, fn))

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=scale,
                           out_threshold=mask_threshold,
                           device=device,
                           labels=labels)
        plt.imshow(mask)
        plt.show()

        plt.imshow(gt_mask)
        plt.show()

        mask = np.array(mask).astype(np.bool)
        result = mask_to_image1D(mask)
        result.save(out_files[i])

        if viz:
            mask = np.array(mask).astype(np.bool)
            colormap = json.load(open(paths.json_file))["colormap"]
            save_img_mask_plot(img,
                               mask_to_image3D(mask, colormap=colormap, paths=paths),
                               ground_truth=gt_mask,
                               fig_name=fn,
                               patient_name=patient,
                               paths=paths)
    if viz:
        img2gif(png_dir=f"{paths.dir_plot_saves}/{patient}",
                target_folder=paths.dir_predicted_gifs,
                out_name=f"{patient}")

    organ_results = eval_inference(patient=patient, mask_dict=mask_dict, paths=paths)
    return organ_results
