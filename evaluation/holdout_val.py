import torch

from evaluation import eval


def holdout(net, writer, logging, optimizer, global_step, imgs, true_masks, val_loader, device, scheduler, masks_pred):
    for tag, value in net.named_parameters():
        tag = tag.replace('.', '/')
        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

    val_score = eval.eval_net(net, val_loader, device)
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
    return val_score
