import logging
import os
import torch

from inference.predict import predict_patient
from training.build_net import build_net
from utilities.data_vis import plot_all_prediction_result

if __name__ == "__main__":

    paths = paths.Paths(db="StructSeg2019_Task3_Thoracic_OAR",
                        model_ckp="Dataset(StructSeg2019_Task3_Thoracic_OAR)_Model(Classic Unet Heart)_Experiment(58)_Epoch(12).pth")
    model = paths.dir_pretrained_model
    scale = 1  # TODO not working now
    mask_threshold = 0.5
    viz = True
    input_size = (1, 512, 512)
    n_classes = 1  # organs+bg
    load_nclasses = 1
    labels=[3]
    dir_test_img = paths.dir_test_img
    models = "Unet"
    metrics = ['Dice', 'Hausdorff Distance 95']

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = build_net(model=models,
                    n_classes=n_classes,
                    data_shape=input_size,
                    sum=False,
                    device=device,
                    load=True,
                    load_dir=model, load_nclasses=load_nclasses)

    results = {}
    for patient in os.listdir(dir_test_img):
        results[patient] = predict_patient(scale=scale,
                                           mask_threshold=mask_threshold,
                                           viz=viz,
                                           patient=patient,
                                           net=net,
                                           device=device, paths=paths, labels=labels)
    e(results, paths=paths, met=metrics)
