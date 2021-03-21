import logging
import os
import torch
from network_architecture.unet import UNet
import paths
from inference.predict import predict_patient

def predict_total():
    model = paths.dir_pretrained_model
    scale = 1
    mask_threshold = 0.4
    save = False
    viz = False

    net = UNet(n_channels=1, n_classes=7)

    logging.info("Loading model {}".format(model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))
    logging.info("Model loaded!")

    for patient in os.listdir(paths.dir_test_img):
        predict_patient(scale=scale, mask_threshold=mask_threshold, save=save, viz=viz, patient=patient, net=net,
                        device=device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    predict_total()