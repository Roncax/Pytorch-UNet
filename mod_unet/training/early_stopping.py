import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_val_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, loss_val, model, path, db_info, train_loss, optimizer, epoch):
        self.path = path

        score = -loss_val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss_val=loss_val, net=model, db_info=db_info, loss=train_loss,
                                 optimizer=optimizer, epoch=epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping on epoch: {epoch}")
        else:
            self.best_score = score
            self.save_checkpoint(loss_val, model, loss=train_loss, optimizer=optimizer, db_info=db_info,
                                 epoch=epoch)
            self.counter = 0

    def save_checkpoint(self, loss_val, net, loss, optimizer, db_info, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.loss_val_min:.6f} --> {loss_val:.6f}).  Saving model ...')

        torch.save({'model_state_dict': net.state_dict(),
                    'train_loss': loss,
                    'val_loss': loss_val,
                    'optimizer_state_dict': optimizer.state_dict()},
                   f=f'{self.path}/'
                     f'Dataset({db_info["name"]})'
                     f'_Model({net.name})'
                     f'_Experiment({db_info["experiments"]})'
                     f'_Epoch({epoch})'
                     f'_Loss({round(loss_val, 4)}).pth')

        self.loss_val_min = loss_val
