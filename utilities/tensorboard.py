import logging
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

import paths


class Board:

    def __init__(self, lr, batch_size, img_scale, epochs):
        self.writer = SummaryWriter(log_dir=f"{paths.dir_tensorboard_runs}/{datetime.now()}",
                                    comment=f'-LR({lr})_BS({batch_size})_SCALE({img_scale})_EPOCHS({epochs})')

    def add_train_values(self, loss, global_step):
        self.writer.add_scalar("Loss/train", loss, global_step)

    def add_validation_values(self, net, global_step, loss_val, optimized_lr, imgs, true_masks, masks_pred):
        self.writer.add_scalar('learning_rate', optimized_lr, global_step)

        if net.n_classes > 1:
            logging.info('Validation cross entropy: {}'.format(loss_val))
            self.writer.add_scalar('Loss/test', loss_val, global_step)
        else:
            logging.info('Validation Dice Coeff: {}'.format(loss_val))
            self.writer.add_scalar('Dice/test', loss_val, global_step)

        self.writer.add_images('images', imgs, global_step)
        if net.n_classes == 1:
            self.writer.add_images('masks/true', true_masks, global_step)
            self.writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

    def add_epoch_results(self, net, global_step):
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
            self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
