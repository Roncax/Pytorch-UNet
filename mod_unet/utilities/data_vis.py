import json
import logging
import os

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from mod_unet.evaluation import metrics
from mod_unet.evaluation.metrics import ConfusionMatrix
from mod_unet.utilities.build_volume import grayscale2rgb_mask


def save_img_mask_plot(img, mask, ground_truth, paths, fig_name="fig", patient_name="Default"):
    save_path = f"{paths.dir_plot_saves}/{patient_name}/{fig_name}"
    fig, ax = plt.subplots(1, 3)

    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].set_title('Output mask')
    ax[1].imshow(mask)
    ax[2].set_title('Ground truth')
    ax[2].imshow(ground_truth)
    ax[1].axis('off')
    ax[2].axis('off')

    os.makedirs(f"{paths.dir_plot_saves}/{patient_name}", exist_ok=True)
    plt.savefig(save_path + ".png")
    plt.close()


def prediction_plot(img, mask, ground_truth):
    fig, ax = plt.subplots(1, 3)

    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[0].axis('off')

    ax[2].set_title('Output mask')
    ax[2].imshow(mask)
    ax[2].axis('off')

    ax[1].set_title('Ground truth')
    ax[1].imshow(ground_truth)
    ax[1].axis('off')

    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return image


# create a gif from images of the target folder
def img2gif(png_dir, target_folder, out_name):
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(target_folder + f"/{out_name}.gif", images)

def volume2gif(volume, target_folder, out_name,):
    imageio.mimsave(f"{target_folder}/{out_name}.gif", volume)


def plot_single_result(score, type, paths, mode, used_net):
    fig, ax = plt.subplots()

    ax.boxplot(x=score.values(), labels=score.keys())
    plt.title(f"{type} {mode}")
    plt.xticks(rotation=-45)
    plt.ylim(bottom=0)

    os.makedirs(f"{paths.dir_plots}", exist_ok=True)
    plt.savefig(paths.dir_plots + f"/{type}_{mode}_{used_net}.png")


def plot_results(results, paths, labels, used_net, met='all', mode=""):
    if met == all:
        met = metrics.ALL_METRICS.keys()

    for m in met:
        plot_single_result(results=results, type=m, paths=paths, labels=labels, mode=mode, used_net=used_net)


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

    plt.show()

def visualize_test(dict_images:dict, info:list=False):
    fontsize = 18

    f, ax = plt.subplots(len(dict_images), figsize=(8, 8))

    i=0
    for img in dict_images.keys():
        ax[i].imshow(dict_images[img])
        ax[i].set_title(img, fontsize=fontsize)
        i+=1
    temp=""
    for str_t in info:
        temp+= " " + str(str_t)
    plt.figtext(0.5, 0.01, temp, ha="center", fontsize=18, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
    plt.show()

# calculate and save all metrics in png
def save_results(results, path_json, met, labels, mode, used_net):
    score = {}
    for organ in labels:
        score[labels[organ]] = []

    logging.info(f"Calculating {type} now")
    with tqdm(total=len(results.keys()), unit='volume') as pbar:
        for patient in results:
            for organ in results[patient]:
                score[organ].append(metrics.ALL_METRICS[type](confusion_matrix=results[patient][organ]))
            pbar.update(1)

    dict_results = json.load(open(path_json))
    dict_results[used_net][mode][met]={}
    for organ in score:
        dict_results[used_net][mode][met][organ]={}
        d = {
            "avg": np.average(score[organ]),
            "min": np.min(score[organ]),
            "max": np.max(score[organ]),
            "25_quantile":np.quantile(score[organ], q=0.25),
            "75_quantile": np.quantile(score[organ], q=0.75)
        }
        dict_results[used_net][mode][met][organ].update(d)


    #open and save metrics (like boxplot) metrics -> organ&info

    return score

def compute_save_metrics(paths, multibin_comb, labels, metrics, used_net=''):
    name = json.load(open(paths.json_file))["name"]
    colormap = json.load(open(paths.json_file))["colormap"]
    results = {}
    mode = "Multibin" if multibin_comb else "Normal"

    with h5py.File(paths.hdf5_results, 'r') as db:
        with h5py.File(paths.hdf5_db, 'r') as db_train:
            with tqdm(total=len(db[f'{name}/test'].keys()), unit='volume') as pbar:
                sample = "volume_46"
                for volume in db[f'{name}/test'].keys():
                    results[volume] = {}
                    vol = []
                    pred_vol = np.empty(shape=(512, 512, 1))
                    gt_vol = np.empty(shape=(512, 512, 1))

                    for slice in sorted(db[f'{name}/test/{volume}/image'].keys(),
                                        key=lambda x: int(x.split("_")[1])):
                        slice_pred_mask = db[f'{name}/test/{volume}/image/{slice}'][()]
                        slice_gt_mask = db_train[f'{name}/test/{volume}/mask/{slice}'][()]
                        slice_test_img = db_train[f'{name}/test/{volume}/image/{slice}'][()]

                        if volume == sample:
                            plot = prediction_plot(img=slice_test_img,
                                                   mask=grayscale2rgb_mask(colormap=colormap, labels=labels,
                                                                           mask=slice_pred_mask),
                                                   ground_truth=grayscale2rgb_mask(colormap=colormap, labels=labels,
                                                                                   mask=slice_gt_mask))

                            vol.append(plot)

                        slice_pred_mask = np.expand_dims(slice_pred_mask, axis=2)
                        pred_vol = np.append(pred_vol, slice_pred_mask, axis=2).astype(dtype=int)

                        slice_gt_mask = np.expand_dims(slice_gt_mask, axis=2)
                        gt_vol = np.append(gt_vol, slice_gt_mask, axis=2).astype(dtype=int)

                    if volume == sample:
                        volume2gif(volume=vol, target_folder=paths.dir_plots, out_name=f"{volume} {mode}")

                    for l in labels.keys():
                        pred_vol_cp = np.zeros(pred_vol.shape)
                        gt_vol_cp = np.zeros(gt_vol.shape)
                        pred_vol_cp[pred_vol == int(l)] = 1
                        gt_vol_cp[gt_vol == int(l)] = 1
                        cm = ConfusionMatrix(test=pred_vol_cp, reference=gt_vol_cp)
                        results[volume][labels[l]] = cm

                    pbar.update(1)

                for m in metrics:
                    results_dict = save_results(results=results, path_json=paths.json_file_inference_results, met=m, labels=labels, mode=mode, used_net=used_net)
                    plot_single_result(score=results_dict, type=m, paths=paths.dir_plots, used_net=used_net, mode=mode)
                    #plot_results(results=i, paths=paths.dir_plots, mode=mode, used_net=used_net)