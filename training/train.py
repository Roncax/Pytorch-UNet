import logging
import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dataset_conversion.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import paths
import numpy as np
from evaluation import eval
from sklearn.model_selection import KFold

from training.early_stopping import EarlyStopping


def train_net(net,
              device,
              epochs,
              batch_size,
              lr,
              val_percent,
              save_cp,
              img_scale, k_folds, patience):
    writer = SummaryWriter(log_dir=f"{paths.dir_tensorboard_runs}/{datetime.now()}",
                           comment=f'-LR({lr})_BS({batch_size})_SCALE({img_scale})_EPOCHS({epochs})')
    global_step = 0

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min' if net.n_classes > 1 else 'max',
                                                     patience=2)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    # kfold = KFold(n_splits=k_folds, shuffle=True)

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()  # weight can be added for class imbalance
    else:
        criterion = nn.BCEWithLogitsLoss()

    loss_min = np.inf

    dataset = BasicDataset(paths.dir_train_imgs, paths.dir_train_masks, img_scale)
    # K-fold Cross Validation model evaluation
    # for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):

    # Divide dataset (1-val_percent) train + val_percent validation
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    # val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                            drop_last=True)

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

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

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
                # epoch_loss += loss.item()

                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % len(train) == 0:
                    loss_val = eval.eval_net(net, val_loader, device)
                    scheduler.step(loss_val)

                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(loss_val))
                        writer.add_scalar('Loss/test', loss_val, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(loss_val))
                        writer.add_scalar('Dice/test', loss_val, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

                    # early_stopping needs the validation loss to check if it has decresed,
                    # and if it has, it will make a checkpoint of the current model
                    if patience != -1:
                        f = f'{paths.dir_checkpoint}/{datetime.now()}_CP-EPOCH({epoch})_LR({lr})_BS({batch_size})_SCALE({img_scale})_EPOCHS({epochs}_VAL_LOSS({loss_val}).pth'
                        early_stopping(loss_val, net, path=f)

                        if early_stopping.early_stop:
                            print(f"Early stopping on epoch: {epoch}")
                            break

                    elif save_cp:
                        if loss_val < loss_min:
                            loss_min = loss_val
                            torch.save(obj=net.state_dict(),
                                       f=f'{paths.dir_checkpoint}/{datetime.now()}_CP-EPOCH({epoch})_LR({lr})_BS({batch_size})_SCALE({img_scale})_EPOCHS({epochs}_VAL_LOSS({loss_val}).pth')
                            logging.info(f'Checkpoint {epoch} saved! Current loss: {loss_val} - Min loss: {loss_min}')

        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

    writer.close()
