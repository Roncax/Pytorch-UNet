from torch import nn

from mod_unet.training.loss import DiceLoss


def build_loss(loss_criterion, n_classes):
    switcher = {
        "dice": DiceLoss(),
        "bce": nn.BCEWithLogitsLoss(),
        "coarse": nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()
    }

    return switcher.get(loss_criterion)