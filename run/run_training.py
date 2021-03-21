import logging
import os
import sys
import numpy as np
import torch
from torch.backends import cudnn
from datetime import datetime
from network_architecture.unet import UNet
from training.train import train_net
import paths
from torchsummary import summary

if __name__ == '__main__':

    load = False
    load_dir = ""  # Load model from a .pth file
    epochs = 1  # Number of epochs
    batch_size = 1 # Batch size
    lr = 0.0001  # Learning rate
    scale = 1  # Downscaling factor of the images
    val = 10.0  # Percent of the databases that is used as validation (0-100)
    save_ckps = True
    deterministic = False
    # TODO
    models = []
    initialization = ''
    optimizer = ''
    loss = ''
    dataset = ''  # vari organi - all
    dropout = ''
    deep_supervision = ''

    # faster convolutions, but more memory
    cudnn.benchmark = True

    if deterministic:
        seed = 123
        cudnn.benchmark = False
        cudnn.deterministic = True
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your databases
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=1, n_classes=7, bilinear=True).cuda()

    summary(net, input_size=(1, 512, 512))

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if load:
        net.load_state_dict(
            torch.load(load_dir, map_location=device)
        )
        logging.info(f'Model loaded from {load_dir}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=epochs,
                  batch_size=batch_size,
                  lr=lr,
                  device=device,
                  img_scale=scale,
                  val_percent=val / 100,
                  save_cp=save_ckps)

    except KeyboardInterrupt:
        torch.save(obj=net.state_dict(), f=f'{paths.dir_checkpoint}/{datetime.now()}_INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
