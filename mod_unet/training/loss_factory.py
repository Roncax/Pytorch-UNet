from torch import nn
from pytorch_toolbelt.losses.dice import DiceLoss
from pytorch_toolbelt.losses.focal import FocalLoss, BinaryFocalLoss


def build_loss(loss_criterion, weight=None):

    switcher = {
        "dice": DiceLoss(mode="binary", smooth=10.0),
        "bce": nn.BCEWithLogitsLoss(),
        "crossentropy": nn.CrossEntropyLoss(weight=weight),
        "binaryFocal": BinaryFocalLoss(),
        "multiclassFocal": FocalLoss()
    }

    return switcher.get(loss_criterion, "Error, the specified criterion doesn't exist")

