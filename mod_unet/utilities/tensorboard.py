
from torch.utils.tensorboard import SummaryWriter

class Board:
    def __init__(self, path, dataset_parameters, active_logs=True):
        self.active_logs = active_logs

        if self.active_logs:
            self.writer = SummaryWriter(log_dir=f'{path}/'
                                                f'{dataset_parameters["name"]}'
                                                f'_experiment({dataset_parameters["experiments"]}).pth')

    def add_train_values(self, loss, global_step):
        if self.active_logs:
            self.writer.add_scalar("Loss/train", loss, global_step)

    def add_validation_values(self, global_step, loss_val, optimized_lr, imgs):
        if self.active_logs:
            self.writer.add_scalar('learning_rate', optimized_lr, global_step)
            self.writer.add_scalar(tag='Loss/test', scalar_value=loss_val, global_step=global_step)
            self.writer.add_images('images', imgs, global_step)

    def add_epoch_results(self, net, global_step):
        if self.active_logs:
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)

                try:
                    self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                except AttributeError:
                    continue
