import torch
from torch import nn
import torch.nn.functional as F

class XentLoss(nn.Module):
    def __init__(self, bWeighted=False, gamma=0, bMask=False):
        """
        categorical (multi-class) cross-entropy loss, capable of adding weight, focal loss.
        input
            bWeighted: True if need weighted loss
            gamma: if >0, becomes focal loss
            bMask: if calculate loss in mask area
        """
        super(XentLoss, self).__init__()
        self.bWeighted = bWeighted
        self.bFL = gamma > 0
        self.gamma = gamma
        self.bMask = bMask
        self.channel_axis = 1

    def forward(self, lb, pred, W=None, mask=None):
        """
        input
            lb: true label, assume to be one-hot format.
            pred: prediction
            W: weight of each class
            mask: one channel tensor of [batch, H, W], elements must be {1,0}
        """
        if self.bMask or self.bWeighted:
            sz = lb.data.shape

        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
        loss = - lb * torch.log(pred)

        # use focal loss
        if self.bFL:  # if use focal loss
            loss = loss * (1 - pred) ** self.gamma
        loss = torch.sum(loss, self.channel_axis)

        # use mask to focus on part of loss
        if self.bMask:
            loss = loss * mask

        # weight the loss
        if self.bWeighted:
            wm = torch.zeros(sz).cuda()
            for k in range(sz[self.channel_axis]):  # for each channel
                wm[:, k, :, :] = lb[:, k, :, :] * W[k]
            loss = loss * torch.sum(wm, self.channel_axis)
        return torch.mean(loss)


# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()
#
#     def forward(self, input, target, W=None):
#         N = target.size(0)
#         smooth = 0.0001
#
#         input = torch.sigmoid(input)
#         print(input)
#
#         input = torch.where(input > 0.5, 1, 0)
#         print(input)
#
#         input_flat = input.view(N, -1)
#         target_flat = target.view(N, -1)
#
#         intersection = input_flat * target_flat
#
#         loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
#         loss = 1 - loss.sum() / N
#
#         return loss


# PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs : torch.Tensor, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)
        inputs = F.logsigmoid(inputs).exp()
        print(inputs)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()

        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        print(dice)
        print(1-dice)
        return 1 - dice


# PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, target, input, W=None):

        C = target.shape[1]

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if W is not None:
                diceLoss *= W[i]
            totalLoss += diceLoss

        if W is not None:
            return totalLoss / sum(W)
        else:
            return totalLoss / (C)