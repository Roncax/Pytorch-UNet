import json
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mod_unet.datasets.hdf5Dataset import HDF5Dataset


# return the predicted mask by higher prob value
# comb_dict = {"0":"name", ..}
def combine_predictions(comb_dict, mask_threshold, shape):
    tot = np.zeros(shape)
    for organ1 in comb_dict:
        t = comb_dict[organ1].copy()
        cd = comb_dict.copy()
        cd.pop(organ1)
        for organ2 in cd:
            t[comb_dict[organ1] < comb_dict[organ2]] = 0

        t[t < mask_threshold] = 0
        t[t > mask_threshold] = 1
        tot[t == 1] = organ1

    return tot


def multibin_prediction(scale, mask_threshold, nets, device, paths, labels, coarse_net):
    dataset = HDF5Dataset(scale=scale, mode='test', db_info=json.load(open(paths.json_file)), paths=paths,
                          labels=labels)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    with h5py.File(paths.hdf5_results, 'w') as db:
        with tqdm(total=len(dataset), unit='img') as pbar:
            for batch in test_loader:
                imgs = batch['image_organ']
                id = batch['id']
                comb_dict = {}

                for organ in nets.keys():
                    nets[organ].eval()
                    img = imgs[organ]


                    assert img.shape[1] == nets[organ].n_channels, \
                        f'Network has been defined with {nets[organ].n_channels} input channels, ' \
                        f'but loaded images have {img.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'
                    img = img.to(device=device, dtype=torch.float32)

                    with torch.no_grad():
                        output = nets[organ](img)

                    probs = torch.sigmoid(output)
                    probs = probs.squeeze(0)
                    full_mask = probs.squeeze().cpu().numpy()
                    # full_mask = full_mask > mask_threshold
                    # res = np.array(full_mask).astype(np.bool)
                    res = full_mask
                    comb_dict[organ] = res

                comb_img = combine_predictions(comb_dict=comb_dict, mask_threshold=mask_threshold, shape=np.shape(res))

                db.create_dataset(id[0], data=comb_img)

                pbar.update(img.shape[0])  # update the pbar by number of imgs in batch
