from torch import nn
from pytorch_toolbelt.losses.dice import DiceLoss
from pytorch_toolbelt.losses.focal import FocalLoss, BinaryFocalLoss
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2


def build_loss(loss_criterion, deep_supervision, weight=None):
    switcher = {
        "dice": DiceLoss(mode="binary", smooth=1.0),
        "bce": nn.BCEWithLogitsLoss(),
        "crossentropy": nn.CrossEntropyLoss(weight=weight),
        "binaryFocal": BinaryFocalLoss(),
        "multiclassFocal": FocalLoss(),
        "dc_bce": BCE_DC_loss()
    }

    loss = switcher.get(loss_criterion, "Error, the specified criterion doesn't exist")

    if deep_supervision:
        loss = MultipleOutputLoss2(loss)

    return loss


class BCE_DC_loss(nn.Module):

    def __init__(self, weight_ce=1, weight_dice=1):
        super(BCE_DC_loss, self).__init__()

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

        self.ce = nn.BCEWithLogitsLoss()
        self.dc = DiceLoss(mode="binary", smooth=1.0)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss

        return result
