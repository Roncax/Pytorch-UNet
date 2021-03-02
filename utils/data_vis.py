import matplotlib.pyplot as plt

save_path = "/home/roncax/Git/Pytorch-UNet/data/Task3_Thoracic_OAR/tests/plt_save/"


def plot_img_and_mask(img, mask, ground_truth, dice=0, fig_name="fig"):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 2)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[0].axis('off')
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
            ax[i + 2].set_title('Ground truth')
            ax[i + 2].imshow(ground_truth[:, :, i])
            ax[i+1].axis('off')
            ax[i + 2].axis('off')
    else:
        ax[1].set_title('Output mask')
        ax[1].imshow(mask)
        ax[2].set_title('Ground truth')
        ax[2].imshow(ground_truth)
        ax[1].axis('off')
        ax[2].axis('off')


    plt.suptitle(f"Dice {dice}", y=0.3)
    plt.savefig(save_path + fig_name + ".png")
    # plt.show()

