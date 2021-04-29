import logging
import torch

from mod_unet.inference.multibin_comb import multibin_prediction
from mod_unet.inference.predict import predict_test_db
from mod_unet.network_architecture.net_factory import build_net
from mod_unet.utilities.paths import Paths
from mod_unet.utilities.data_vis import compute_viz_metrics


# labels = {"0": "Bg",
#           "1": "RightLung",
#           "2": "LeftLung",
#           "3": "Heart",
#           "4": "Trachea",
#           "5": "Esophagus",
#           "6": "SpinalCord"
#           }


if __name__ == "__main__":
    db_name = "StructSeg2019_Task3_Thoracic_OAR"
    scale = 1  # TODO not working now

    load_dir_list = {
        "1": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet_FineTuning_RightLung)_Experiment(293)_Epoch(4)_Loss(0.0385).pth",
        "2": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet_FineTuning_LeftLung)_Experiment(298)_Epoch(8).pth_Loss(0.0318).pth",
        "3": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet_FineTuning_Heart)_Experiment(299)_Epoch(13).pth_Loss(0.0619).pth",
        "4": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet_FineTuning_Trachea)_Experiment(302)_Epoch(15)_Loss(0.2132).pth",
        "5": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet_FineTuning_Esophagus)_Experiment(303)_Epoch(10)_Loss(0.1941).pth",
        "6": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet_FineTuning_SpinalCord)_Experiment(304)_Epoch(8)_Loss(0.1045).pth",
        "coarse": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet Coarse)_Experiment(458)_Epoch(14)_Loss(0.0094)_LossCrit(coarse)_Scale(0.7)_augmentedDB.pth"
    }
    mask_threshold = 0.5
    viz = True
    input_size = (1, 512, 512)
    models = "Unet"
    metrics = ['Dice', 'Hausdorff Distance 95']
    data_shape = (1, 512, 512)
    multibin_comb = False
    labels = {"0": "Bg",
              "1": "RightLung",
              "2": "LeftLung",
              "3": "Heart",
              "4": "Trachea",
              "5": "Esophagus",
              "6": "SpinalCord"
              }
    n_classes = 1 if len(labels) == 2 else len(labels)  # class number in net -> #classes+1(Bg)
    old_classes = 7

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if multibin_comb:
        nets = {}
        labels.pop("0")
        for label in labels.keys():
            paths = Paths(db=db_name,
                          model_ckp=load_dir_list[label])
            nets[label] = build_net(model=models,
                                    n_classes=1,
                                    data_shape=input_size,
                                    device=device,
                                    load_inference=True,
                                    load_dir=paths.dir_pretrained_model)
            nets[label].name += f" {labels[label]}"
            logging.info(f"Network {nets[label].name} active")

        paths = Paths(db=db_name,
                      model_ckp=load_dir_list["coarse"])
        coarse_net = build_net(model=models,
                        n_classes=n_classes,
                        data_shape=input_size,
                        device=device,
                        load_inference=True,
                        load_dir=paths.dir_pretrained_model)

        paths = Paths(db=db_name)
        multibin_prediction(scale=scale, labels=labels, mask_threshold=mask_threshold, device=device, paths=paths,
                            nets=nets, coarse_net=coarse_net)

    else:
        labels.pop("0")
        paths = Paths(db=db_name,
                      model_ckp=load_dir_list["coarse"])
        net = build_net(model=models,
                        n_classes=n_classes,
                        data_shape=input_size,
                        device=device,
                        load_inference=True,
                        load_dir=paths.dir_pretrained_model)

        # predict all images in test folder
        predict_test_db(labels=labels, mask_threshold=mask_threshold, device=device, net=net,
                        scale=scale, paths=paths)

        if viz:
            compute_viz_metrics(paths=paths, multibin_comb=multibin_comb, labels=labels, metrics=metrics)