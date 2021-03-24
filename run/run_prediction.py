import logging
import os
import torch
from network_architecture.unet import UNet
import paths
from inference.predict import predict_patient
from training.build_net import build_net


def predict_total():
    model = paths.dir_pretrained_model
    scale = 1
    mask_threshold = 0.5
    save = True
    viz = True
    input_size = (1, 512, 512)
    n_classes = 7  # organs+bg
    dir_test_img = paths.dir_test_img
    models = ['Unet']

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    nets = build_net(models=models, n_classes=n_classes, data_shape=input_size, sum=False)

    logging.info("Loading model {}".format(model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #TODO different net compare (for now only Unet)
    for net in nets:
        nets[net].to(device=device)
        nets[net].load_state_dict(torch.load(model, map_location=device))
        logging.info("Model loaded!")

        for patient in os.listdir(dir_test_img):
            predict_patient(scale=scale,
                            mask_threshold=mask_threshold,
                            save=save,
                            viz=viz,
                            patient=patient,
                            net=nets[net],
                            device=device)


if __name__ == "__main__":
    predict_total()
