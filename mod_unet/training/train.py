import json
import logging
import time
from statistics import mean

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from mod_unet.datasets.hdf5Dataset import HDF5Dataset
from torch.utils.data import DataLoader, random_split
from mod_unet.evaluation import eval
from mod_unet.utilities.data_vis import visualize_test

from mod_unet.training.loss.loss_factory import build_loss
from mod_unet.training.early_stopping import EarlyStopping
from mod_unet.utilities.tensorboard import Board


def train_net(net, device, epochs, batch_size,
              lr, val_percent, img_scale,
              patience, paths, labels,
              loss_criterion, augmentation, deep_supervision,
              dict_results, dict_db_parameters, start_time,debug_mode=False):
    global_step = 0
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=2)

    criterion = build_loss(loss_criterion=loss_criterion)  # TODO diceloss che non funge

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # DATASET split train/val
    dataset = HDF5Dataset(scale=img_scale, mode='train', db_info=dict_db_parameters, paths=paths,
                          labels=labels, augmentation=augmentation)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                            drop_last=True)

    # tensorboard viz
    tsboard = Board(dataset_parameters=dataset.db_info, path=paths.dir_tensorboard_runs, active_logs=not debug_mode)

    logging.info(f'''Starting:
        Net:             {net.name}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Train samples:   {len(train)}
        Val samples:     {len(val)}
        Images scaling:  {img_scale}
        Labels:          {len(labels)}
    ''')

    # main loop
    for epoch in range(epochs):
        net.train()  # net in train mode
        loss_list = []
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

                # different number of losses returned in deep_supervision
                if deep_supervision:
                    if net.name == "Unet" or net.name=="SE-ResUnet":
                        y0, y1, y2, y3, y4 = net(imgs)
                        loss0 = criterion(y0, true_masks.squeeze(1) if net.n_classes > 1 else true_masks)
                        loss1 = criterion(y1, true_masks.squeeze(1) if net.n_classes > 1 else true_masks)
                        loss2 = criterion(y2, true_masks.squeeze(1) if net.n_classes > 1 else true_masks)
                        loss3 = criterion(y3, true_masks.squeeze(1) if net.n_classes > 1 else true_masks)
                        loss4 = criterion(y4, true_masks.squeeze(1) if net.n_classes > 1 else true_masks)
                        loss = loss0 + loss1 + loss2 + loss3 + loss4



                        if global_step % 50 == 0:

                            net.eval()
                            dict={
                                "img_real":imgs.squeeze().cpu().detach().numpy(),
                                "gt":true_masks.squeeze().cpu().detach().numpy(),
                                "prediction":y0.squeeze().cpu().detach().numpy()*255
                            }
                            info = []
                            info.append(loss0.item())

                            visualize_test(dict_images=dict, info=info)
                            net.train()

                    if net.name== "NestedUnet":
                        y0, y1, y2, y3 = net(imgs)
                        loss0 = criterion(y0, true_masks.squeeze(1) if net.n_classes > 1 else true_masks)
                        loss1 = criterion(y1, true_masks.squeeze(1) if net.n_classes > 1 else true_masks)
                        loss2 = criterion(y2, true_masks.squeeze(1) if net.n_classes > 1 else true_masks)
                        loss3 = criterion(y3, true_masks.squeeze(1) if net.n_classes > 1 else true_masks)
                        loss = loss0 + loss1 + loss2 + loss3

                else:
                    masks_pred = net(imgs)
                    loss = criterion(masks_pred, true_masks.squeeze(1) if net.n_classes > 1 else true_masks)

                loss_list.append(loss.item())
                if global_step % 10 == 0:
                    tsboard.add_train_values((loss0 if deep_supervision else loss).item(), global_step)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])  # update the pbar by number of imgs in batch
                global_step += 1

                # validation
                if global_step % int(len(train)) == 0:
                    loss_val = eval.eval_train(net, val_loader, device, deep_supervision=deep_supervision)
                    scheduler.step(loss_val)
                    tsboard.add_validation_values(global_step=global_step, loss_val=loss_val,
                                                  optimized_lr=optimizer.param_groups[0]['lr'], imgs=imgs)

                    path_name = f'{paths.dir_checkpoint}/' \
                                f'Dataset({dataset.db_info["name"]})' \
                                f'_Experiment({dataset.db_info["experiments"]})' \
                                f'_Epoch({epoch})'

                    if not debug_mode:
                        temp_dict={"validation_loss": loss_val, "avg_train_loss": mean(loss_list),"elapsed_time": time.time()-start_time}
                        dict_results[dataset.db_info["experiments"]]["epochs"][str(epoch)] = {}
                        dict_results[dataset.db_info["experiments"]]["epochs"][str(epoch)].update(temp_dict)
                        json.dump(dict_results, open(paths.json_file_train_results, "w"))

                        # early_stopping needs the validation loss to check if it has decreased,
                        # and if it has, it will make a checkpoint of the current model
                        if patience != -1:
                            early_stopping(loss_val=loss_val, model=net,
                                           train_loss=loss0 if deep_supervision else loss, optimizer=optimizer,
                                           epoch=epoch, file_name=path_name + '.pth')
                            if early_stopping.early_stop:
                                break

                        else:
                            torch.save({'model_state_dict': net.state_dict(),
                                        'train_loss': loss0 if deep_supervision else loss,
                                        'val_loss': loss_val,
                                        'optimizer_state_dict': optimizer.state_dict()},
                                       f=path_name + '.pth')

        tsboard.add_epoch_results(net=net, global_step=global_step)
        if early_stopping.early_stop: break

    tsboard.writer.close()
