import logging

import torch
from torch.utils.tensorboard import SummaryWriter

class Board:

    def __init__(self, path, net, dataset_parameters):
        self.writer = SummaryWriter(log_dir=f'{path}/'
                                            f'{net.name}'
                                            f'-{dataset_parameters["name"]}'
                                            f'-experiment({dataset_parameters["experiments"]}).pth')

    def add_train_values(self, loss, global_step):
        self.writer.add_scalar("Loss/train", loss, global_step)
        #self.writer.add_scalars(main_tag="Loss/test-train", tag_scalar_dict={"Train": loss}, global_step=global_step)

    def add_validation_values(self, net, global_step, loss_val, optimized_lr, imgs):
        self.writer.add_scalar('learning_rate', optimized_lr, global_step)

        if net.n_classes > 1:
            logging.info('Validation cross entropy: {}'.format(loss_val))
            self.writer.add_scalar(tag='Loss/test', scalar_value=loss_val, global_step=global_step)

            #self.writer.add_scalars(main_tag="Loss/test-train", tag_scalar_dict={"Test":loss_val}, global_step=global_step)
        else:
            logging.info('Validation Dice Coeff: {}'.format(loss_val))
            self.writer.add_scalar('Dice/test', loss_val, global_step)


        self.writer.add_images('images', imgs, global_step)


    def add_epoch_results(self, net, global_step):
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
            self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
