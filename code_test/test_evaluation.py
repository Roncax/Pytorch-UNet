import os
import paths
from evaluation.metrics import ConfusionMatrix
import evaluation.metrics as metrics
from utilities.various import build_np_volume


for patient in os.listdir(paths.dir_test_img):
    patient_volume_gt = build_np_volume(dir = os.path.join(paths.dir_test_GTimg, patient))
    patient_volume_pred = build_np_volume(dir = os.path.join(paths.dir_mask_prediction, patient))

    cm = ConfusionMatrix(test=patient_volume_pred, reference=patient_volume_gt)
    print(cm.get_matrix())
    print(metrics.dice(confusion_matrix=cm))






