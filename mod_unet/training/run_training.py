import logging
import sys
import numpy as np
import torch
from torch.backends import cudnn
from datetime import datetime

from mod_unet.network_architecture.net_factory import build_net
from mod_unet.training.train import train_net
from torchsummary import summary

from mod_unet.utilities.paths import Paths

epochs = 1000  # Number of epochs
batch_size = 1  # Batch size
lr = 0.0001  # Learning rate
scale = 0.7  # Downscaling factor of the images
val = 0.2  # Databases that is used as validation (0-1)
deterministic = False  # deterministic results, but slower
patience = 5  # -1 -> no early stopping, save all
finetuning = False
feature_extraction = False
model = "Unet"  # net type
verbose = True
val_round_freq = 1  # every val_round*train_len images there is a validation round
data_shape = (1, 512, 512)
old_classes = 7  # if finetuning or fe, specify the old class number
augmentation = True
labels = {"0": "Bg",
          "1": "RightLung",
          "2": "LeftLung",
          "3": "Heart",
          "4": "Trachea",
          "5": "Esophagus",
          "6": "SpinalCord"
          }
# For reference
# labels = {"0": "Bg",
#           "1": "RightLung",
#           "2": "LeftLung",
#           "3": "Heart",
#           "4": "Trachea",
#           "5": "Esophagus",
#           "6": "SpinalCord"
#           }
n_classes = 1 if len(labels) == 2 else len(labels)  # class number in net -> #classes+1(Bg)
multi_binary = False

db_name = "StructSeg2019_Task3_Thoracic_OAR"
load_dir_list = {
    "1": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet Coarse)_Experiment(256)_Epoch(15)_ValLoss(0.007305324633466566).pth",
    "2": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet Coarse)_Experiment(256)_Epoch(15)_ValLoss(0.007305324633466566).pth",
    "3": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet Coarse)_Experiment(256)_Epoch(15)_ValLoss(0.007305324633466566).pth",
    "4": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet Coarse)_Experiment(256)_Epoch(15)_ValLoss(0.007305324633466566).pth",
    "5": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet Coarse)_Experiment(256)_Epoch(15)_ValLoss(0.007305324633466566).pth",
    "6": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet Coarse)_Experiment(256)_Epoch(15)_ValLoss(0.007305324633466566).pth",
    "coarse": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet Coarse)_Experiment(256)_Epoch(15)_ValLoss(0.007305324633466566).pth"
}

loss_criterion = {"0": "Bg",
                  "1": "coarse",
                  "2": "coarse",
                  "3": "coarse",
                  "4": "coarse",
                  "5": "coarse",
                  "6": "coarse",
                  "coarse":"coarse"
                  }

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

    if not multi_binary:
        paths = Paths(db=db_name,
                      model_ckp=load_dir_list["coarse"])
        net = build_net(model=model,
                        n_classes=n_classes,
                        finetuning=finetuning,
                        load_dir=paths.dir_pretrained_model,
                        device=device,
                        data_shape=data_shape, old_classes=old_classes, feature_extraction=feature_extraction, verbose=verbose)
        net.name += " Coarse"

        try:
            train_net(net=net,
                      epochs=epochs,
                      batch_size=batch_size,
                      lr=lr,
                      device=device,
                      img_scale=scale,
                      val_percent=val,
                      patience=patience,
                      val_round=val_round_freq,
                      paths=paths, labels=labels,
                      loss_criterion=loss_criterion['coarse'],
                      augmentation=augmentation)

        except KeyboardInterrupt:
            torch.save(obj=net.state_dict(), f=f'{paths.dir_checkpoint}/{datetime.now()}_INTERRUPTED.pth')
            logging.info('Saved interrupt')
            sys.exit(0)

    else:
        lab = filter(lambda x: x > 0, list(map(int, labels.keys())))

        for l in lab:
            label = {str(l): labels[str(l)]}
            paths = Paths(db=db_name,
                          model_ckp=load_dir_list[str(l)])
            net = build_net(model=model,
                            n_classes=1,
                            finetuning=finetuning,
                            load_dir=paths.dir_pretrained_model,
                            device=device,
                            data_shape=data_shape, old_classes=old_classes, feature_extraction=feature_extraction, verbose=verbose)

            try:
                train_net(net=net,
                          epochs=epochs,
                          batch_size=batch_size,
                          lr=lr,
                          device=device,
                          img_scale=scale,
                          val_percent=val,
                          patience=patience,
                          val_round=val_round_freq,
                          paths=paths,
                          labels=label,
                          loss_criterion=loss_criterion[str(l)],
                          augmentation=augmentation)

            except KeyboardInterrupt:
                torch.save(obj=net.state_dict(), f=f'{paths.dir_checkpoint}/{datetime.now()}_INTERRUPTED.pth')
                logging.info('Saved interrupt')
                sys.exit(0)
