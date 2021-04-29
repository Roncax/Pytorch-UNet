import json
import logging

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from mod_unet.datasets.hdf5Dataset import HDF5Dataset
from torch.utils.data import DataLoader, random_split
from mod_unet.evaluation import eval

from mod_unet.training.loss_factory import build_loss
from mod_unet.training.early_stopping import EarlyStopping
from mod_unet.utilities.tensorboard import Board

def train_net(net, device, epochs, batch_size,
              lr, val_percent, img_scale,
              patience, val_round, paths, labels,
              loss_criterion, augmentation):
    global_step = 0
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=2)

    # TODO diceloss che non funge
    criterion = build_loss(loss_criterion=loss_criterion, n_classes=net.n_classes)

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # DATASET split train/val
    dataset = HDF5Dataset(scale=img_scale, mode='train', db_info=json.load(open(paths.json_file)), paths=paths,
                          labels=labels, augmentation=augmentation)
    if augmentation:
        pass

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                            drop_last=True)

    # tensorboard viz
    tsboard = Board(dataset_parameters=dataset.db_info, net=net, path=paths.dir_tensorboard_runs, active_logs=True)

    logging.info(f'''Starting:
        Net:             {net.name}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Train samples:   {len(train)}
        Val samples:     {len(val)}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Labels:          {len(labels)}
        Patience:        {patience}
    ''')

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

                loss = criterion(masks_pred, true_masks.squeeze(1) if net.n_classes > 1 else true_masks)

                if global_step % 10 == 0:
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

                    path_name = f'{paths.dir_checkpoint}/' \
                                f'Dataset({dataset.db_info["name"]})' \
                                f'_Model({net.name})' \
                                f'_Experiment({dataset.db_info["experiments"]})' \
                                f'_Epoch({epoch})' \
                                f'_Loss({round(loss_val, 4)})' \
                                f'_LossCrit({loss_criterion})' \
                                f'_Scale({img_scale})'

                    if augmentation:
                        path_name += '_augmentedDB'

                    # early_stopping needs the validation loss to check if it has decreased,
                    # and if it has, it will make a checkpoint of the current model
                    if patience != -1:
                        early_stopping(loss_val=loss_val, model=net,
                                       train_loss=loss, optimizer=optimizer, epoch=epoch, file_name=path_name+'.pth')
                        if early_stopping.early_stop:
                            break

                    else:
                        torch.save({'model_state_dict': net.state_dict(),
                                    'train_loss': loss,
                                    'val_loss': loss_val,
                                    'optimizer_state_dict': optimizer.state_dict()},
                                   f=path_name+'.pth')

        tsboard.add_epoch_results(net=net, global_step=global_step)
        if early_stopping.early_stop: break

    tsboard.writer.close()
