import torch
import torch.nn.functional as F
from tqdm import tqdm
from evaluation import metrics


def eval_net(net, loader, device):
    """Evaluation of the net (multiclass -> crossentropy, binary -> dice)"""
    net.eval()  # the net is in evaluation mode
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batches
    tot = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False, miniters=1) as pbar:
        # iterate over all val batch
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            # iterate over all files of single batch
            for true_mask, pred in zip(true_masks, mask_pred):
                if net.n_classes > 1:
                    # multiclass evaluation over single image-mask pair
                    tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask).item()

                else:
                    # Single class evaluation over all validation volume
                    pred = (pred > 0.5).float()  # 0 or 1 by threeshold
                    pred = pred.detach().cpu().numpy()
                    cm = metrics.ConfusionMatrix(test=pred, reference=true_mask)
                    tp_, fp_, tn_, fn_ = cm.get_matrix()
                    tp += tp_
                    fp += fp_
                    tn += tn_
                    fn += fn_

            pbar.update(n=1)

    net.train()  # the net return to training mode
    return 2 * tp / (2 * tp + fp + fn) if net.n_classes == 1 else tot / n_val
