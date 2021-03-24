import logging

from torchsummary import summary

from network_architecture.unet import UNet


# create a net for every specified model
def build_net(models, data_shape, n_classes, sum):
    nets = {}
    switcher = {
        "Unet": build_Unet(data_shape=data_shape, n_classes=n_classes, model="Unet")
    }

    for model in models:
        nets[model] = switcher.get(model)
        if sum:
            summary(nets[model], input_size=data_shape)

    assert (len(nets) > 0), "No models"

    return nets


def build_Unet(data_shape, n_classes, model):
    net = UNet(n_channels=data_shape[0], n_classes=n_classes, bilinear=True).cuda()
    logging.info(f'''Network {model} uploaded:
                {net.n_channels} input channels
                {net.n_classes} output channels (classes)
                {"Bilinear" if net.bilinear else "Transposed conv"} upscaling
                ''')
    return net
