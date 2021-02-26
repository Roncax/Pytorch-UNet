import torch
import torch.nn.functional as F
from tqdm import tqdm


def eval_net(net, loader, device):
    net.eval()
    n_val = len(loader)  # the number of batches
    score = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.no_grad():
                mask_pred = net(imgs)

            score += F.cross_entropy(mask_pred, true_masks).item()
            pbar.update()

    net.train()
    return score / n_val
