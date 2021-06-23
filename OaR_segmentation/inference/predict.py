import json
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from OaR_segmentation.utilities.build_volume import volume_mask_to_1Darray
from OaR_segmentation.datasets.hdf5Dataset import HDF5Dataset


def predict_test_db(scale, mask_threshold, net, paths, labels):
    net.eval()
    dataset = HDF5Dataset(scale=scale, mode='test', db_info=json.load(open(paths.json_file_database)), paths=paths,
                          labels=labels, channels=net.n_channels)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    with h5py.File(paths.hdf5_results, 'w') as db:
        with tqdm(total=len(dataset), unit='img') as pbar:
            for batch in test_loader:
                imgs = batch['image'].to(device="cuda", dtype=torch.float32)
                id = batch['id']

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

                if net.n_classes > 1:
                    res = volume_mask_to_1Darray(full_mask)
                else:
                    res = full_mask

                db.create_dataset(id[0], data=res)
                pbar.update(imgs.shape[0])  # update the pbar by number of imgs in batch

