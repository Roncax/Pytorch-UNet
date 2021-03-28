import logging
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from dataset_conversion.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from evaluation import eval

from training.early_stopping import EarlyStopping
from training.build_optimizer import build_optimizer
from training.loss_criterion import build_loss_criterion
from utilities.tensorboard import Board


def train_net(net,
              device,
              epochs,
              batch_size,
              lr,
              val_percent,
              save_cp,
              img_scale,
              patience,
              optimizer_mode,
              loss_mode,
              val_round, paths):
    global_step = 0
    optimizer = build_optimizer(mode=optimizer_mode, net=net, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min' if net.n_classes > 1 else 'max',
                                                     patience=2)
    criterion = build_loss_criterion(loss_mode, net)
    early_stopping = EarlyStopping(patience=patience, verbose=True,
                                   path=paths.dir_checkpoint)  # initialize the early_stopping object

    # DATASET
    dataset = BasicDataset(paths=paths, scale=img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                            drop_last=True)

    tsboard = Board(dataset_parameters=dataset.dataset_parameters, net=net,
                    path=paths.dir_tensorboard_runs)  # for tensorboard viz

    logging.info(f'''Starting:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Train samples:   {len(train)}
        Val samples:     {len(val)}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    breaker = False  # used for early stopping
    # main loop
    for epoch in range(epochs):
        net.train()  # net in train mode

        # batches loop
        with tqdm(total=len(train), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)

                loss = criterion(masks_pred, true_masks.squeeze(1))

                tsboard.add_train_values(loss.item(), global_step)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])  # update the pbar by number of imgs in batch
                global_step += 1

                # validation
                if global_step % int(len(train) * val_round) == 0:
                    loss_val = eval.eval_train(net, val_loader, device)
                    scheduler.step(loss_val)

                    tsboard.add_validation_values(net=net, global_step=global_step, loss_val=loss_val,
                                                  optimized_lr=optimizer.param_groups[0]['lr'], imgs=imgs)

                    # early_stopping needs the validation loss to check if it has decresed,
                    # and if it has, it will make a checkpoint of the current model
                    if patience != -1:
                        early_stopping(loss_val, net, path=paths.dir_checkpoint, ds=dataset.dataset_parameters,
                                       train_loss=loss, loss_mode=loss_mode, optimizer=optimizer, epoch=epoch)

                        if early_stopping.early_stop:
                            print(f"Early stopping on epoch: {epoch}")
                            breaker = True
                            break

                    else:
                        torch.save({'model_state_dict': net.state_dict(),
                                    'train_loss': loss,
                                    'val_loss': loss_val,
                                    'loss_mode': loss_mode,
                                    'optimizer_state_dict': optimizer.state_dict()},
                                   f=f'{paths.dir_checkpoint}/'
                                     f'Dataset({dataset.dataset_parameters["name"]})'
                                     f'_Model({net.name})'
                                     f'_Experiment({dataset.dataset_parameters["experiments"]})'
                                     f'_Epoch({epoch}).pth')
        # needed for early stopping
        if breaker:
            break
        tsboard.add_epoch_results(net=net, global_step=global_step)
    tsboard.writer.close()
