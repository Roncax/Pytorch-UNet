import logging
import sys
import numpy as np
import torch
from torch.backends import cudnn
from datetime import datetime

from mod_unet.network_architecture.net_factory import build_net
from mod_unet.training.train import train_net
from mod_unet.utilities import paths
from torchsummary import summary

epochs = 100  # Number of epochs
batch_size = 1  # Batch size
lr = 0.0001  # Learning rate
scale = 0.1  # Downscaling factor of the images
val = 0.2  # Databases that is used as validation (0-1)
save_ckpts = True  # save all ckpts
deterministic = False  # deterministic results, but slower
patience = 5  # -1 -> no early stopping
finetuning = False
model = "Unet"  # net type
net_summary = False  # summary of all the models
val_round_freq = 1  # every val_round*train_len images there is a validation round
paths = paths.Paths(db="StructSeg2019_Task3_Thoracic_OAR",
                    model_ckp="")
data_shape = (1, 512, 512)
finetuning_ckp_classes = 7  # if finetuning, specify the old class number
labels = {"0": "Bg",
          "1": "RightLung",
          "2": "LeftLung",
          "3": "Heart",
          "4": "Trachea",
          "5": "Esophagus",
          "6": "SpinalCord"
          }
n_classes = 1 if len(labels) == 2 else len(labels)  # class number in net -> #classes+1(Bg)

if __name__ == '__main__':
    cudnn.benchmark = True  # faster convolutions, but more memory
    if deterministic:
        seed = 1234
        cudnn.benchmark = False
        cudnn.deterministic = True
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = build_net(model,
                    n_classes=n_classes,
                    finetuning=finetuning,
                    load_dir=paths.dir_pretrained_model,
                    device=device,
                    data_shape=data_shape,
                    load_nclasses=finetuning_ckp_classes)
    if n_classes > 1:
        net.name += " Coarse"
    else:
        foo = filter(lambda x: x > 0, list(map(int, labels.keys())))
        net.name += f" {labels[str(list(foo)[0])]}"

    if net_summary: summary(net, input_size=data_shape)

    try:
        train_net(net=net,
                  epochs=epochs,
                  batch_size=batch_size,
                  lr=lr,
                  device=device,
                  img_scale=scale,
                  val_percent=val,
                  save_cp=save_ckpts,
                  patience=patience,
                  val_round=val_round_freq,
                  paths=paths, labels=labels)

    except KeyboardInterrupt:
        torch.save(obj=net.state_dict(), f=f'{paths.dir_checkpoint}/{datetime.now()}_INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
