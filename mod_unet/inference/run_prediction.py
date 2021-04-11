import json
import logging
import h5py
import numpy as np
import torch
from tqdm import tqdm

from mod_unet.inference.predict import predict_test_db
from mod_unet.network_architecture.net_factory import build_net
from mod_unet.utilities.paths import Paths
from mod_unet.utilities.data_vis import plot_results, prediction_plot, volume2gif
from mod_unet.utilities.build_volume import grayscale2rgb_mask
from mod_unet.evaluation.metrics import ConfusionMatrix

if __name__ == "__main__":
    paths = Paths(db="StructSeg2019_Task3_Thoracic_OAR",
                  model_ckp="Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet Coarse)_Experiment(249)_Epoch(0).pth_ValLoss(0.027069566008305377).pth")
    scale = 1  # TODO not working now
    mask_threshold = 0.5
    viz = True
    input_size = (1, 512, 512)
    models = "Unet"
    metrics = ['Dice', 'Hausdorff Distance 95']
    data_shape = (1, 512, 512)
    labels = {"0": "Bg",
              "1": "RightLung",
              "2": "LeftLung",
              "3": "Heart",
              "4": "Trachea",
              "5": "Esophagus",
              "6": "SpinalCord"
              }
    n_classes = 1 if len(labels) == 2 else len(labels)  # class number in net -> #classes+1(Bg)

    # labels = {"0": "Bg",
    #           "1": "RightLung",
    #           "2": "LeftLung",
    #           "3": "Heart",
    #           "4": "Trachea",
    #           "5": "Esophagus",
    #           "6": "SpinalCord"
    #           }

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = build_net(model=models,
                    n_classes=n_classes,
                    data_shape=input_size,
                    device=device,
                    finetuning=True,
                    load_dir=paths.dir_pretrained_model, load_nclasses=n_classes)

    # predict all images in test folder
    # predict_test_db(labels=labels, mask_threshold=mask_threshold, device=device, net=net,
    #                 scale=scale, paths=paths)

    if viz:
        name = json.load(open(paths.json_file))["name"]
        colormap = json.load(open(paths.json_file))["colormap"]
        results = {}

        with h5py.File(paths.hdf5_results, 'r') as db:
            with h5py.File(paths.hdf5_db, 'r') as db_train:
                with tqdm(total=len(db[f'{name}/test'].keys()), unit='volume') as pbar:
                    for volume in db[f'{name}/test'].keys():
                        results[volume] = {}
                        vol = []
                        pred_vol = np.empty(shape=(512, 512, 1))
                        gt_vol = np.empty(shape=(512, 512, 1))
                        for slice in db[f'{name}/test/{volume}/image'].keys():
                            slice_pred_mask = db[f'{name}/test/{volume}/image/{slice}'][()]
                            slice_gt_mask = db_train[f'{name}/test/{volume}/mask/{slice}'][()]
                            slice_test_img = db_train[f'{name}/test/{volume}/image/{slice}'][()]

                            plot = prediction_plot(img=slice_test_img,
                                                   mask=grayscale2rgb_mask(colormap=colormap, labels=labels,
                                                                           mask=slice_pred_mask),
                                                   ground_truth=grayscale2rgb_mask(colormap=colormap, labels=labels,
                                                                                   mask=slice_gt_mask))

                            vol.append(plot)

                            slice_pred_mask = np.expand_dims(slice_pred_mask, axis=2)
                            pred_vol = np.append(pred_vol, slice_pred_mask, axis=2).astype(dtype=int)

                            slice_gt_mask = np.expand_dims(slice_gt_mask, axis=2)
                            gt_vol = np.append(gt_vol, slice_gt_mask, axis=2).astype(dtype=int)

                        volume2gif(volume=vol, target_folder=paths.dir_plots, out_name=volume)

                        for l in labels.keys():
                            pred_vol_cp = np.zeros(pred_vol.shape)
                            gt_vol_cp = np.zeros(gt_vol.shape)
                            pred_vol_cp[pred_vol == int(l)] = 1
                            gt_vol_cp[gt_vol == int(l)] = 1
                            results[volume][labels[l]] = ConfusionMatrix(test=pred_vol, reference=gt_vol)

                        pbar.update(1)
                    plot_results(results=results, paths=paths, met=metrics)
