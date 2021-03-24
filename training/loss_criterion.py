from torch import nn


def build_loss_criterion(mode, net):
    multiclass_switcher = {
        "CrossEntropyLoss": CrossEntropyLoss()
    }

    binary_switcher = {
        "BCEWithLogitsLoss": BCEWithLogitsLoss()
    }

    if net.n_classes > 1:
        return multiclass_switcher.get(mode, "No multiclass optimizer")
    else:
        return binary_switcher.get(mode, "No binary optimizer")


def CrossEntropyLoss():
    criterion = nn.CrossEntropyLoss()
    return criterion


def BCEWithLogitsLoss():
    criterion = nn.BCEWithLogitsLoss()
    return criterion
