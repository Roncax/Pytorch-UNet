import json
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

def volume2gif(volume, target_folder, out_name):
    imageio.mimsave(f"{target_folder}/{out_name}.gif", volume)


def plot_single_result(results, type, paths):
    fig, ax = plt.subplots()

    with open(paths.json_file) as f:
        mask_dict = json.load(f)["labels"]

    score = {}
    for organ in mask_dict:
        score[mask_dict[organ]] = []

    with tqdm(total=results.keys(), unit='volume') as pbar:
        for patient in results:
            for organ in results[patient]:
                score[organ].append(metrics.ALL_METRICS[type](confusion_matrix=results[patient][organ]))
            pbar.update(1)

    ax.boxplot(x=score.values(), labels=score.keys())
    plt.title(type)
    plt.xticks(rotation=-45)

    os.makedirs(f"{paths.dir_plots}", exist_ok=True)
    plt.savefig(paths.dir_plots + f"/{type}.png")

    plt.show()


def plot_results(results, paths, met='all'):
    if met == all:
        met = metrics.ALL_METRICS.keys()

    for m in met:
        plot_single_result(results=results, type=m, paths=paths)
