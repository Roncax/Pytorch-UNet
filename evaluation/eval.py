import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluation import metrics


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()  # the net is in evaluation mode
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:

        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()

                pred = pred.detach().cpu().numpy()
                true_masks = true_masks.detach().cpu().numpy()
                cm = metrics.ConfusionMatrix(test=pred, reference=true_masks)
                tp_, fp_, tn_, fn_ = cm.get_matrix()
                tp += tp_
                fp += fp_
                tn += tn_
                fn += fn_

                # tot += dice_coeff(pred, true_masks).item()
            pbar.set_postfix(**{'loss (batch)': tp})
            pbar.update()
        tot += 2 * tp / (2 * tp + fp + fn)
    net.train()  # the net return to training mode
    return tot
