import sys

sys.path.append(r'/home/roncax/Git/Pytorch-UNet/') # /content/gdrive/MyDrive/Colab/Thesis_OaR_Segmentation/

from OaR_segmentation.utilities.paths import Paths
import json
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from OaR_segmentation.network_architecture.net_factory import build_net
from OaR_segmentation.training.custom_trainer_stacking import CustomTrainer_stacking

from OaR_segmentation.datasets.hdf5Dataset import HDF5Dataset




def create_combined_dataset(scale, mask_threshold, nets,  paths, labels):
    dataset = HDF5Dataset(scale=scale, mode='test', db_info=json.load(open(paths.json_file_database)), paths=paths, channels=1,
                          labels=labels, stacking=True)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    i=0
    with h5py.File(paths.hdf5_stacking, 'w') as db:
        with tqdm(total=len(dataset), unit='img') as pbar:
            for batch in test_loader:
                imgs = batch['mask_dict']
                mask_gt = batch['mask_gt']
                

                for organ in nets.keys():
                    nets[organ].eval()
                    img = imgs[organ].to(device="cuda", dtype=torch.float32)

                    #logits
                    with torch.no_grad():
                        output = nets[organ](img)

                    #probs = torch.sigmoid(output)
                    probs = output.squeeze(0)

                    full_mask = probs.squeeze().cpu().numpy()
                    
                    # full_mask = full_mask > mask_threshold
                    # res = np.array(full_mask).astype(np.bool)
                    #res = full_mask
                    db.create_dataset(f"{i}/{organ}", data=full_mask)


                mask_gt = mask_gt.to(device="cuda", dtype=torch.float32)
                mask_gt=mask_gt.squeeze().cpu().numpy()
                db.create_dataset(f"{i}/gt", data=mask_gt)
    
                i +=1

                pbar.update(img.shape[0])  # update the pbar by number of imgs in batch

        
    
def stacking_training(paths, labels, platform):
    loss_criterion = 'crossentropy'
    lr = 1e-4

    net = build_net(model='unet', n_classes=7, channels=7, load_inference=False)
    
    
    trainer = CustomTrainer_stacking( paths=paths, image_scale=1, augmentation=False,
                            batch_size=1, loss_criterion=loss_criterion, val_percent=0.5,
                            labels=labels, network=net, deep_supervision=False, dropout=False,
                            fine_tuning=False, feature_extraction=False,
                            pretrained_model='', lr=lr, patience=5, epochs=1000,
                            multi_loss_weights=[1,1], platform=platform)

    trainer.initialize()
    trainer.run_training()   
    
    
    
if __name__=="__main__":
    db_name = "StructSeg2019_Task3_Thoracic_OAR"
    platform = "local" #local, colab, polimi


    load_dir_list = {
        "1": "733/Dataset(StructSeg2019_Task3_Thoracic_OAR)_Experiment(733)_Epoch(15).pth",
        "2": "734/Dataset(StructSeg2019_Task3_Thoracic_OAR)_Experiment(734)_Epoch(12).pth",
        "3": "735/Dataset(StructSeg2019_Task3_Thoracic_OAR)_Experiment(735)_Epoch(3).pth",
        "4": "736/Dataset(StructSeg2019_Task3_Thoracic_OAR)_Experiment(736)_Epoch(17).pth",
        "5": "737/Dataset(StructSeg2019_Task3_Thoracic_OAR)_Experiment(737)_Epoch(15).pth",
        "6": "738/Dataset(StructSeg2019_Task3_Thoracic_OAR)_Experiment(738)_Epoch(13).pth",
        "coarse": "1043/model_best.model"
    }
    models = {"1": "seresunet",
              "2": "seresunet",
              "3": "seresunet",
              "4": "seresunet",
              "5": "seresunet",
              "6": "seresunet",
              "coarse": "unet"
              }
    deeplabv3_backbone = "mobilenet"  # resnet, drn, mobilenet, xception

    labels = {
        "0": "Bg",
        "1": "RightLung",
        "2": "LeftLung",
        "3": "Heart",
        "4": "Trachea",
        "5": "Esophagus",
        "6": "SpinalCord"
    }
    n_classes = len(labels) if len(labels) > 2 else 1
    scale = 1
    mask_threshold = 0.5
    channels = 1
    lr = 1e-4

    paths = Paths(db=db_name, platform=platform)


    labels.pop("0")  # don't want to predict also the background
    
    
    nets = {}
    for label in labels.keys():
        paths.set_pretrained_model(load_dir_list[label])

        nets[label] = build_net(model=models[label], n_classes=1, channels=channels, load_inference=True,
                                load_dir=paths.dir_pretrained_model)
    
    #create_combined_dataset(nets=nets,scale=scale, paths=paths, labels=labels, mask_threshold=mask_threshold )
    
    stacking_training(paths=paths, labels=labels, platform=platform)