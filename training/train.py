import logging
import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from datetime import datetime
from evaluation.eval import eval_net
from torch.utils.tensorboard import SummaryWriter
from dataset_conversion.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import paths
from evaluation.holdout_val import holdout
import numpy as np


def train_net(net,
              device,
              epochs,
              batch_size,
              lr,
              val_percent,
              save_cp,
              img_scale):
    dataset = BasicDataset(paths.dir_train_imgs, paths.dir_train_masks, img_scale)

    # Divide dataset (1-val_percent) train + val_percent validation
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(log_dir=f"{paths.dir_tensorboard_runs}/_{datetime.now()}",
                           comment=f'-LR({lr})_BS({batch_size})_SCALE({img_scale})_EPOCHS({epochs})')
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min' if net.n_classes > 1 else 'max',
                                                     patience=2)

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()  # weight can be added for class imbalance
        print("Crossentropy")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("BCEWithLogitsLoss")

    loss_min = 99999
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
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

                if (global_step == 1):
                    writer.add_graph(net, imgs)

                masks_pred = net(imgs)

                # print(f"image img shape: {imgs.size()}")
                # print(f"image prediction shape: {masks_pred.size()}")
                # print(f"mask shape: {true_masks.size()}")
                # print(masks_pred)
                # print(f"mask shape: {true_masks.squeeze(1).size()}")

                torch.set_printoptions(threshold=10_000)

                loss = criterion(masks_pred, true_masks.squeeze(1))
                epoch_loss += loss.item()

                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % (len(dataset) // (10 * batch_size)) == 0:
                    loss_current = holdout(net=net, writer=writer, logging=logging, optimizer=optimizer, global_step=global_step,
                                           imgs=imgs,
                                           true_masks=true_masks, val_loader=val_loader, device=device, scheduler=scheduler,
                                           masks_pred=masks_pred)

        if save_cp:
            if loss_current < loss_min:
                torch.save(obj=net.state_dict(),
                           f=f'{paths.dir_checkpoint}/{datetime.now()}_CP_EPOCH{epoch}-LR({lr})_BS({batch_size})_SCALE({img_scale})_EPOCHS({epochs}_VAL_LOSS({loss_current}).pth')
                logging.info(f'Checkpoint {epoch} saved! Current loss: {loss_current} - Min loss: {loss_min}')

    writer.close()
