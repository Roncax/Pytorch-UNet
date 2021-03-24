from torch import optim


def build_optimizer(mode, net, lr):
    switcher = {
        "rmsprop": rmsprop(net=net, lr=lr)
    }

    return switcher.get(mode, "No optimizer")


def rmsprop(net, lr):
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    return optimizer
