import json
import logging

import h5py
import numpy as np
import torch
from tqdm import tqdm

from mod_unet.inference.multibin_comb import multibin_prediction
from mod_unet.inference.predict import predict_test_db
from mod_unet.network_architecture.net_factory import build_net
from mod_unet.utilities.build_volume import grayscale2rgb_mask
from mod_unet.utilities.data_vis import prediction_plot, volume2gif, plot_single_result
from mod_unet.utilities.paths import Paths
from mod_unet.evaluation import metrics
from mod_unet.evaluation.metrics import ConfusionMatrix

def compute_save_metrics(paths, labels, metrics_list, db_name, colormap, experiment_num,
                         sample_gif_name="volume_46", shape=(512, 512, 1)):
    # colormap = json.load(open(paths.json_file))["colormap"]
    results = {}

    with h5py.File(paths.hdf5_results, 'r') as db:
        with h5py.File(paths.hdf5_db, 'r') as db_train:
            with tqdm(total=len(db[f'{db_name}/test'].keys()), unit='volume') as pbar:
                for volume in db[f'{db_name}/test'].keys():
                    results[volume] = {}
                    vol = []
                    pred_vol = np.empty(shape=shape)
                    gt_vol = np.empty(shape=shape)

                    for slice in sorted(db[f'{db_name}/test/{volume}/image'].keys(),
                                        key=lambda x: int(x.split("_")[1])):
                        slice_pred_mask = db[f'{db_name}/test/{volume}/image/{slice}'][()]
                        slice_gt_mask = db_train[f'{db_name}/test/{volume}/mask/{slice}'][()]
                        slice_test_img = db_train[f'{db_name}/test/{volume}/image/{slice}'][()]

                        if volume == sample_gif_name:
                            msk = grayscale2rgb_mask(colormap=colormap, labels=labels, mask=slice_pred_mask)
                            gt = grayscale2rgb_mask(colormap=colormap, labels=labels, mask=slice_gt_mask)
                            plot = prediction_plot(img=slice_test_img, mask=msk, ground_truth=gt)
                            vol.append(plot)

                        slice_pred_mask = np.expand_dims(slice_pred_mask, axis=2)
                        pred_vol = np.append(pred_vol, slice_pred_mask, axis=2).astype(dtype=int)

                        slice_gt_mask = np.expand_dims(slice_gt_mask, axis=2)
                        gt_vol = np.append(gt_vol, slice_gt_mask, axis=2).astype(dtype=int)

                    if volume == sample_gif_name:
                        volume2gif(volume=vol, target_folder=paths.dir_plots, out_name=f"example({volume})_inference({experiment_num})")

                    for l in labels.keys():
                        pred_vol_cp = np.zeros(pred_vol.shape)
                        gt_vol_cp = np.zeros(gt_vol.shape)
                        pred_vol_cp[pred_vol == int(l)] = 1
                        gt_vol_cp[gt_vol == int(l)] = 1
                        cm = ConfusionMatrix(test=pred_vol_cp, reference=gt_vol_cp)
                        results[volume][labels[l]] = cm

                    pbar.update(1)

                for m in metrics_list:
                    results_dict = save_results(results=results, path_json=paths.json_file_inference_results, met=m,
                                                experiment_num=experiment_num, test_info=dict_test_info, labels=labels)

                    plot_single_result(score=results_dict, type=m, paths=paths.dir_plots, exp_num=experiment_num)


# calculate and save all metrics in png
def save_results(results, path_json, met, labels, experiment_num, test_info):
    dict_results = json.load(open(path_json))
    dict_results[experiment_num] = test_info
    dict_results["num"] = experiment_num
    dict_results[experiment_num][met] = {}
    score = {}
    for organ in labels:
        score[labels[organ]] = []

    logging.info(f"Calculating {met} now")
    with tqdm(total=len(results.keys()), unit='volume') as pbar:
        for patient in results:
            for organ in results[patient]:
                score[organ].append(metrics.ALL_METRICS[met](confusion_matrix=results[patient][organ]))
            pbar.update(1)

    for organ in score:
        d = {
            "data":score[organ],
            "avg": np.average(score[organ]),
            "min": np.min(score[organ]),
            "max": np.max(score[organ]),
            "25_quantile": np.quantile(score[organ], q=0.25),
            "75_quantile": np.quantile(score[organ], q=0.75)
        }
        dict_results[experiment_num][met][organ] = d

    json.dump(dict_results, open(path_json, "w"))
    return score


if __name__ == "__main__":
    db_name = "StructSeg2019_Task3_Thoracic_OAR"

    load_dir_list = {
        "1": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(SE-ResUnet_RightLung DeepSupervision)_Experiment(530)_Epoch(26)_Loss(0.0393)_LossCrit(coarse)_Scale(1)_augmentedDB.pth",
        "2": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(SE-ResUnet_LeftLung DeepSupervision)_Experiment(531)_Epoch(11)_Loss(0.0354)_LossCrit(coarse)_Scale(1)_augmentedDB.pth",
        "3": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet_FineTuning_Heart)_Experiment(470)_Epoch(10)_Loss(0.0732)_LossCrit(coarse)_Scale(1)_augmentedDB.pth",
        "4": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet_FineTuning_Trachea)_Experiment(471)_Epoch(9)_Loss(0.4676)_LossCrit(coarse)_Scale(1)_augmentedDB.pth",
        "5": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet_FineTuning_Esophagus)_Experiment(472)_Epoch(4)_Loss(0.5256)_LossCrit(coarse)_Scale(1)_augmentedDB.pth",
        "6": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet_FineTuning_SpinalCord)_Experiment(473)_Epoch(8)_Loss(0.2333)_LossCrit(coarse)_Scale(1)_augmentedDB.pth",
        "coarse": "Dataset(StructSeg2019_Task3_Thoracic_OAR)_Experiment(664)_Epoch(23).pth"
    }

    scale = 1
    mask_threshold = 0.5
    viz = True
    input_size = (1, 512, 512)
    models = {"1": "SE-ResUnet",
              "2": "SE-ResUnet",
              "3": "Unet",
              "4": "Unet",
              "5": "Unet",
              "6": "Unet",
              "coarse":"Unet"
              }
    metrics_list = ['Dice', 'Hausdorff Distance 95']
    multibin_comb = True
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

    paths = Paths(db=db_name, model_ckp=load_dir_list["coarse"])
    dict_inference_results = json.load(open(paths.json_file_inference_results))
    dict_db_info = json.load(open(paths.json_file_database))
    experiment_num = dict_inference_results["num"] + 1

    # used for results storage purpose
    dict_test_info = {
        "db": db_name,
        "used_models": load_dir_list,
        "scale": scale,
        "mask_threshold": mask_threshold,
        "models": models,
        "multibinary_combination": multibin_comb,
        "labels": labels
    }

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    labels.pop("0")  # don't want to predict also the background

    # BEGIN INFERENCE
    coarse_net = build_net(model=models["coarse"], n_classes=n_classes, data_shape=input_size,
                           device=device, load_inference=True, load_dir=paths.dir_pretrained_model)

    if multibin_comb:
        nets = {}
        for label in labels.keys():
            paths_temp = Paths(db=db_name, model_ckp=load_dir_list[label])
            nets[label] = build_net(model=models[label], n_classes=1, data_shape=input_size,
                                    device=device, load_inference=True,
                                    load_dir=paths_temp.dir_pretrained_model)

            logging.info(f"Network for {labels[label]} prediction active")

        multibin_prediction(scale=scale, labels=labels, mask_threshold=mask_threshold, device=device,
                            paths=paths, nets=nets, coarse_net=coarse_net)

    else:
        predict_test_db(labels=labels, mask_threshold=mask_threshold, device=device, net=coarse_net,
                        scale=scale, paths=paths)

    # METRICS CALCULATION
    compute_save_metrics(paths=paths, labels=labels, metrics_list=metrics_list,db_name=db_name, colormap=dict_db_info["colormap"],experiment_num=experiment_num)


