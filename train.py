import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.backends import cudnn
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_checkpoint = 'checkpoints/'

dir_img = "/home/roncax/Git/Pytorch-UNet/data/Task3_Thoracic_OAR/Thoracic_OAR_img/images"
dir_mask = "/home/roncax/Git/Pytorch-UNet/data/Task3_Thoracic_OAR/Thoracic_OAR_img/masks"

def train_net(net,
              device,
              epochs,
              batch_size,
              lr,
              val_percent,
              save_cp,
              img_scale):
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # Divide dataset (1-val_percent) train + val_percent validation
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'-LR({lr})_BS({batch_size})_SCALE({img_scale})_EPOCHS({epochs})')
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
    else:
        criterion = nn.BCEWithLogitsLoss()

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

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                # evaluation step, update the learning rate and check if better DICE is archieved
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_EPOCH{epoch + 1}-LR({lr})_BS({batch_size})_SCALE({img_scale})_EPOCHS({epochs}).pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


if __name__ == '__main__':

    load = False
    load_dir = "/home/roncax/Git/Pytorch-UNet/checkpoints/CP_EPOCH1-LR(0.0001)_BS(3)_SCALE(1)_EPOCHS(1).pth"  # Load model from a .pth file
    epochs = 1  # Number of epochs
    batch_size = 3  # Batch size
    lr = 0.0001  # Learning rate
    scale = 1  # Downscaling factor of the images
    val = 10.0  # Percent of the data that is used as validation (0-100)
    save_ckps = True
    deterministic = True

    # faster convolutions, but more memory
    cudnn.benchmark = True

    if deterministic:
        seed = 123
        cudnn.benchmark = False
        cudnn.deterministic = True
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=1, n_classes=1, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if load:
        net.load_state_dict(
            torch.load(load_dir, map_location=device)
        )
        logging.info(f'Model loaded from {load_dir}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=epochs,
                  batch_size=batch_size,
                  lr=lr,
                  device=device,
                  img_scale=scale,
                  val_percent=val / 100,
                  save_cp=save_ckps)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
