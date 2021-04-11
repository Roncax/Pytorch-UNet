import logging

import torch

from mod_unet.network_architecture.unet import UNet
from mod_unet.network_architecture.unet.unet_parts import OutConv


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


# create a net for every specified model
def build_net(model, data_shape, n_classes, device, finetuning=False, load_dir=None, feature_extraction=False,
              load_nclasses=None):
    switcher = {
        "Unet": build_Unet(data_shape=data_shape, n_classes=n_classes, model="Unet", finetuning=finetuning, load_dir=load_dir,
                           device=device, feature_extraction=feature_extraction, load_nclasses=load_nclasses)
    }

    return switcher.get(model)


def build_Unet(data_shape, n_classes, model, finetuning, load_dir, device, feature_extraction, load_nclasses):
    classes = load_nclasses if finetuning else n_classes

    net = UNet(n_channels=data_shape[0], n_classes=classes, bilinear=True).cuda()
    logging.info(f'''Network {model} uploaded:
                {net.n_channels} input channels
                {net.n_classes} output channels (classes)
                {"Bilinear" if net.bilinear else "Transposed conv"} upscaling
                Fine tuning: {finetuning}
                Feature extraction: {feature_extraction}
                ''')

    if finetuning:
        net.name += "_FineTuning"
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        logging.info(f'Model {net.name} loaded from {load_dir} for fine tuning')

    elif feature_extraction:
        net.name += "_FeatureExtraction"
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        set_parameter_requires_grad(net)
        net.outc = OutConv(64, n_classes)
        logging.info(f'Model {net.name} loaded from {load_dir} for feature extraction')

    net.to(device=device)

    return net
