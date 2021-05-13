import logging

import torch
import torchvision.models.segmentation
from torch import nn

from mod_unet.network_architecture.unet import UNet
from mod_unet.network_architecture.se_resunet import SeResUNet


from mod_unet.network_architecture.unet import OutConv
from mod_unet.network_architecture.se_resunet import outconv
from mod_unet.network_architecture.nestedUnet_model import NestedUNet

from torchsummary import summary


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


# create a net for every specified model
def build_net(model, data_shape, n_classes, device, finetuning=False, load_dir=None, feature_extraction=False,
              old_classes=None, load_inference=False, verbose=False, dropout=False, deep_supervision=False):
    if model == "Unet":
        net = build_Unet(data_shape=data_shape, n_classes=n_classes, finetuning=finetuning, load_dir=load_dir,
                         device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                         load_inference=load_inference, verbose=verbose, deep_supervision=deep_supervision)
    elif model == "SE-ResUnet":
        net = build_SeResUNet(data_shape=data_shape, n_classes=n_classes, finetuning=finetuning,
                              load_dir=load_dir,
                              device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                              load_inference=load_inference, verbose=verbose, dropout=dropout,
                              deep_supervision=deep_supervision),
    elif model == "NestedUnet":
        net = build_NestedUNet(data_shape=data_shape, n_classes=n_classes, finetuning=finetuning,
                               load_dir=load_dir,
                               device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                               load_inference=load_inference, verbose=verbose, deep_supervision=deep_supervision),
    elif model == "deeplabv3_resnet50":
        net = torchvision.models.segmentation.segmentation.deeplabv3_resnet50(num_classes=(2 if n_classes==1 else n_classes)).to(device),

    else:
        net = None

    assert net is not None

    return net


def build_Unet(data_shape, n_classes, finetuning, load_dir, device, feature_extraction, old_classes, load_inference,
               verbose, deep_supervision):
    if finetuning:
        net = UNet(n_channels=data_shape[0], n_classes=old_classes, bilinear=True,
                   deep_supervision=deep_supervision).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        net.outc = OutConv(64, n_classes)
        net.n_classes = n_classes

    elif feature_extraction:
        net = UNet(n_channels=data_shape[0], n_classes=old_classes, bilinear=True,
                   deep_supervision=deep_supervision).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        set_parameter_requires_grad(net)
        net.outc = OutConv(64, n_classes, )
        net.n_classes = n_classes

    elif load_inference:
        net = UNet(n_channels=data_shape[0], n_classes=n_classes, bilinear=True).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        net.n_classes = n_classes
    else:
        net = UNet(n_channels=data_shape[0], n_classes=n_classes, bilinear=True,
                   deep_supervision=deep_supervision).cuda()

    net.to(device=device)

    if verbose: summary(net, input_size=data_shape)

    return net


def build_SeResUNet(data_shape, n_classes, finetuning, load_dir, device, feature_extraction, old_classes,
                    load_inference, verbose, dropout, deep_supervision):
    if finetuning:
        net = SeResUNet(n_channels=data_shape[0], n_classes=old_classes, deep_supervision=deep_supervision,
                        dropout=dropout).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        net.outc = outconv(64, n_classes, dropout=True, rate=0.1)
        net.n_classes = n_classes

    elif feature_extraction:
        net = SeResUNet(n_channels=data_shape[0], n_classes=old_classes, deep_supervision=deep_supervision,
                        dropout=dropout).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        set_parameter_requires_grad(net)
        net.outc = outconv(64, n_classes, dropout=True, rate=0.1)
        net.n_classes = n_classes

    elif load_inference:
        net = SeResUNet(n_channels=data_shape[0], n_classes=n_classes, deep_supervision=False, dropout=False).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        net.n_classes = n_classes
    else:
        net = SeResUNet(n_channels=data_shape[0], n_classes=n_classes, deep_supervision=deep_supervision,
                        dropout=dropout).cuda()

    net.to(device=device)

    if verbose: summary(net, input_size=data_shape)

    return net


def build_NestedUNet(data_shape, n_classes, finetuning, load_dir, device, feature_extraction, old_classes,
                     load_inference, verbose, deep_supervision):
    if finetuning:
        net = NestedUNet(input_channels=data_shape[0], num_classes=old_classes,
                         deep_supervision=deep_supervision).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        net.final = nn.Conv2d(32, n_classes, kernel_size=1)
        net.n_classes = n_classes

    elif feature_extraction:
        net = NestedUNet(input_channels=data_shape[0], num_classes=old_classes,
                         deep_supervision=deep_supervision).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        set_parameter_requires_grad(net)
        net.final = nn.Conv2d(32, n_classes, kernel_size=1)
        net.n_classes = n_classes

    elif load_inference:
        net = NestedUNet(input_channels=data_shape[0], num_classes=n_classes, deep_supervision=False).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        net.n_classes = n_classes
    else:
        net = NestedUNet(input_channels=data_shape[0], num_classes=n_classes, deep_supervision=deep_supervision).cuda()

    net.to(device=device)

    if verbose: summary(net, input_size=data_shape)

    return net
