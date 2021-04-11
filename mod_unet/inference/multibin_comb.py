import json
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mod_unet.utilities.build_volume import volume_mask_to_1Darray
from mod_unet.datasets.hdf5Dataset import HDF5Dataset


def multibin_prediction(scale, mask_threshold, nets, device, paths, labels):
    dataset = HDF5Dataset(scale=scale, mode='test', db_info=json.load(open(paths.json_file)), paths=paths,
                          labels=labels)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)



    with tqdm(total=len(dataset), unit='img') as pbar:
        for batch in test_loader:
            imgs = batch['image']
            id = batch['id']

            assert imgs.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            imgs = imgs.to(device=device, dtype=torch.float32)

            img2comb = {}
            for net in nets:
                net.eval()

                with torch.no_grad():
                    output = net(imgs)

                if net.n_classes > 1:
                    probs = F.softmax(output, dim=1)  # prob from 0 to 1 (dim = masks)
                else:
                    probs = torch.sigmoid(output)

                probs = probs.squeeze(0)
                full_mask = probs.squeeze().cpu().numpy()
                full_mask = full_mask > mask_threshold

                full_mask = np.array(full_mask).astype(np.bool)
                res = volume_mask_to_1Darray(full_mask)


            with h5py.File(paths.hdf5_results, 'a') as db:
                db.create_dataset(id[0], data=res)

            pbar.update(imgs.shape[0])  # update the pbar by number of imgs in batch
