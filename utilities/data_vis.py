import os

import imageio
import matplotlib.pyplot as plt
from utilities.various import check_create_dir
import paths


def plot_img_and_mask(img, mask, ground_truth, dice=0, fig_name="fig", patient_name="Default"):
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

    plt.suptitle(f"Dice {dice}", y=0.3)

    check_create_dir(f"{paths.dir_plot_saves}/{patient_name}")
    plt.savefig(save_path + ".png")
    plt.close()

# create a gif from images of the target folder
def img2gif(png_dir, target_folder, out_name):
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(target_folder + f"{out_name}.gif", images)