import logging

import torch

from mod_unet.network_architecture.deeplab_v3p import DeepLab
from mod_unet.network_architecture.segnet import SegNet
from mod_unet.network_architecture.unet import UNet
from mod_unet.network_architecture.se_resunet import SeResUNet

from mod_unet.network_architecture.unet import OutConv
from mod_unet.network_architecture.se_resunet import outconv

from torchsummary import summary


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


# create a net for every specified model
def build_net(model, channels, n_classes, finetuning=False, load_dir=None, feature_extraction=False,
              old_classes=None, load_inference=False, dropout=False, deep_supervision=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model == "Unet":
        net = build_Unet(channels=channels, n_classes=n_classes, finetuning=finetuning, load_dir=load_dir,
                         device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                         load_inference=load_inference, deep_supervision=deep_supervision)
    elif model == "SE-ResUnet":
        net = build_SeResUNet(channels=channels, n_classes=n_classes, finetuning=finetuning,
                              load_dir=load_dir,
                              device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                              load_inference=load_inference, dropout=dropout,
                              deep_supervision=deep_supervision)
    # TODO sistemare net builder completo
    elif model == "segnet":
        net = SegNet(input_nbr=1, label_nbr=n_classes).cuda()
        net.name = "SegNet"
        net.n_classes = n_classes
        net.n_channels = channels
    elif model == "deeplabv3":
        net = DeepLab(backbone='resnet', output_stride=16, num_classes=n_classes).cuda()
        net.name = "DeepLab V3"
        net.n_classes = n_classes
        net.n_channels = channels

    else:
        net=None
        print("WARNING! The specified net doesn't exist")

    return net


def build_Unet(channels, n_classes, finetuning, load_dir, device, feature_extraction, old_classes, load_inference,
               deep_supervision):
    if finetuning or feature_extraction:
        net = UNet(n_channels=channels, n_classes=old_classes, bilinear=True,
                   deep_supervision=deep_supervision).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['state_dict'])
        if feature_extraction:
            set_parameter_requires_grad(net)
        net.outc = OutConv(64, n_classes)

    elif load_inference:
        net = UNet(n_channels=channels, n_classes=n_classes, bilinear=True).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['state_dict'])

    else:
        net = UNet(n_channels=channels, n_classes=n_classes, bilinear=True,
                   deep_supervision=deep_supervision).cuda()

    net.n_classes = n_classes

    return net.to(device=device)


def build_SeResUNet(channels, n_classes, finetuning, load_dir, device, feature_extraction, old_classes,
                    load_inference, dropout, deep_supervision):
    if finetuning or feature_extraction:
        net = SeResUNet(n_channels=channels, n_classes=old_classes, deep_supervision=deep_supervision,
                        dropout=dropout).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['state_dict'])
        if feature_extraction:
            set_parameter_requires_grad(net)
        net.outc = outconv(64, n_classes, dropout=True, rate=0.1)

    elif load_inference:
        net = SeResUNet(n_channels=channels, n_classes=n_classes, deep_supervision=False, dropout=False).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['state_dict'])

    else:
        net = SeResUNet(n_channels=channels, n_classes=n_classes, deep_supervision=deep_supervision,
                        dropout=dropout).cuda()

    net.n_classes = n_classes

    return net.to(device=device)
