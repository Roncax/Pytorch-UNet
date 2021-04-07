import logging

import torch
from torchsummary import summary

from network_architecture.unet import UNet
from network_architecture.unet.unet_parts import OutConv


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


# create a net for every specified model
def build_net(model, data_shape, n_classes, sum, device, load=False, load_dir=None, feature_extraction=True, load_nclaasses=None):
    switcher = {
        "Unet": build_Unet(data_shape=data_shape, n_classes=n_classes, model="Unet", finetuning=load, load_dir=load_dir,
                           device=device, feature_extraction=feature_extraction, load_nclasses=load_nclaasses)
    }

    net = switcher.get(model)
    if sum:
        summary(net, input_size=data_shape)

    return net


def build_Unet(data_shape, n_classes, model, finetuning, load_dir, device, feature_extraction, load_nclasses):
    classes = load_nclasses if finetuning else n_classes

    net = UNet(n_channels=data_shape[0], n_classes=classes, bilinear=True).cuda()
    logging.info(f'''Network {model} uploaded:
                {net.n_channels} input channels
                {net.n_classes} output channels (classes)
                {"Bilinear" if net.bilinear else "Transposed conv"} upscaling
                ''')

    mod = ""
    if finetuning:
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        mod = "_FineTuning"
        logging.info(f'Model {net.name} loaded from {load_dir}')
        if feature_extraction:
            mod = "_FeatureExtraction"
            set_parameter_requires_grad(net)
            net.outc = OutConv(64, n_classes)

    net.name = net.name + mod
    net.to(device=device)

    return net
