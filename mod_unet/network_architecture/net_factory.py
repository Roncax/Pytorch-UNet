import logging

import torch

from mod_unet.network_architecture.unet import UNet
from mod_unet.network_architecture.unet.unet_parts import OutConv


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


# create a net for every specified model
def build_net(model, data_shape, n_classes, device, finetuning=False, load_dir=None, feature_extraction=False,
              old_classes=None, load_inference=False):
    switcher = {
        "Unet": build_Unet(data_shape=data_shape, n_classes=n_classes, finetuning=finetuning, load_dir=load_dir,
                           device=device, feature_extraction=feature_extraction,old_classes=old_classes,load_inference=load_inference)
    }

    return switcher.get(model)


def build_Unet(data_shape, n_classes, finetuning, load_dir, device, feature_extraction, old_classes, load_inference):

    net = UNet(n_channels=data_shape[0], n_classes=n_classes, bilinear=True).cuda()

    if finetuning:
        net = UNet(n_channels=data_shape[0], n_classes=old_classes, bilinear=True).cuda()
        net.name += "_FineTuning"
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        net.outc = OutConv(64, n_classes)
        net.n_classes = n_classes

    elif feature_extraction:
        net = UNet(n_channels=data_shape[0], n_classes=old_classes, bilinear=True).cuda()
        net.name += "_FeatureExtraction"
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        set_parameter_requires_grad(net)
        net.outc = OutConv(64, n_classes)
        net.n_classes = n_classes

    elif load_inference:
        net = UNet(n_channels=data_shape[0], n_classes=n_classes, bilinear=True).cuda()
        net.name += "_Inference"
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        net.n_classes = n_classes

    logging.info(f'''Network Unet uploaded:
                {net.n_channels} input channels
                {net.n_classes} output channels (classes)
                {"Bilinear" if net.bilinear else "Transposed conv"} upscaling
                Fine tuning: {finetuning} with {old_classes} classes
                Feature extraction: {feature_extraction} with {old_classes} classes
                ''')


    net.to(device=device)

    return net
