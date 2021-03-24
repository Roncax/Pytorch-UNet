import logging
import sys
import numpy as np
import torch
from torch.backends import cudnn
from datetime import datetime

from training.build_net import build_net
from training.train import train_net
import paths

if __name__ == '__main__':

    load = False
    load_dir = paths.dir_pretrained_model  # Load model from a .pth file
    epochs = 100  # Number of epochs
    batch_size = 1  # Batch size
    lr = 0.0001  # Learning rate
    scale = 1  # Downscaling factor of the images
    val = 0.2  # Databases that is used as validation (0-1)
    save_ckps = False
    deterministic = False  # deterministic results, but slower
    patience = 5  # =-1 no early stopping
    n_classes = 7
    data_shape = (1, 512, 512)
    models = ["Unet"]
    summary = True  # summary of all the models?
    val_round = 1  # every val_round*train_len images there is a validation round
    loss_mode= "CrossEntropyLoss"
    optimizer_mode = "rmsprop"

    cudnn.benchmark = True  # faster convolutions, but more memory

    # TODO add below parameters and logic
    initialization = ''
    optimizer = ''
    loss = ''
    dataset = ''  # vari organi - all
    dropout = ''
    deep_supervision = ''
    k_folds = 5

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

    nets = build_net(models, data_shape, n_classes, sum=summary)

    # train all the specified nets
    for net in nets:
        # TODO different loads based on model
        if load:
            nets[net].load_state_dict(
                torch.load(load_dir, map_location=device)
            )
            logging.info(f'Model loaded from {load_dir}')

        nets[net].to(device=device)

        try:
            train_net(net=nets[net],
                      epochs=epochs,
                      batch_size=batch_size,
                      lr=lr,
                      device=device,
                      img_scale=scale,
                      val_percent=val,
                      save_cp=save_ckps,
                      patience=patience,
                      val_round=val_round,
                      loss_mode=loss_mode,
                      optimizer_mode=optimizer_mode)

        except KeyboardInterrupt:
            torch.save(obj=net.state_dict(), f=f'{paths.dir_checkpoint}/{datetime.now()}_INTERRUPTED.pth')
            logging.info('Saved interrupt')
            sys.exit(0)
