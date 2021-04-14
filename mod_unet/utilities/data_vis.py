import json
import logging
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from mod_unet.evaluation import metrics


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


def plot_single_result(results, type, paths, labels, mode):
    fig, ax = plt.subplots()

    score = {}
    for organ in labels:
        score[labels[organ]] = []

    logging.info(f"Calculating {type} now")
    with tqdm(total=len(results.keys()), unit='volume') as pbar:
        for patient in results:
            for organ in results[patient]:
                score[organ].append(metrics.ALL_METRICS[type](confusion_matrix=results[patient][organ]))
            pbar.update(1)

    ax.boxplot(x=score.values(), labels=score.keys())
    plt.title(f"{type} {mode}")
    plt.xticks(rotation=-45)

    os.makedirs(f"{paths.dir_plots}", exist_ok=True)
    plt.savefig(paths.dir_plots + f"/{type}_{mode}.png")


def plot_results(results, paths, labels, met='all', mode=""):
    if met == all:
        met = metrics.ALL_METRICS.keys()

    for m in met:
        plot_single_result(results=results, type=m, paths=paths, labels=labels, mode=mode)
