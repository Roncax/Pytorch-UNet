import logging
import os
import torch
import paths
from inference.predict import predict_patient
from training.build_net import build_net
from utilities.data_vis import plot_all_prediction_result


def predict_total():
    model = paths.dir_pretrained_model
    scale = 1 # TODO not working now
    mask_threshold = 0.5
    viz = True
    input_size = (1, 512, 512)
    n_classes = 7  # organs+bg
    dir_test_img = paths.dir_test_img
    models = ['Unet']
    metrics = ['Dice', 'Jaccard', 'Hausdorff Distance 95', 'Accuracy']

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    nets = build_net(models=models, n_classes=n_classes, data_shape=input_size, sum=False, device=device, load=True, load_dir=model)

    # TODO different net comparison (for now only Unet)
    for net in nets:

        results = {}
        for patient in os.listdir(dir_test_img):
            results[patient] = predict_patient(scale=scale,
                                               mask_threshold=mask_threshold,
                                               viz=viz,
                                               patient=patient,
                                               net=nets[net],
                                               device=device)
        plot_all_prediction_result(results, metrics)


if __name__ == "__main__":
    predict_total()
