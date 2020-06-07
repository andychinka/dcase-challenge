import numpy as np
import torch

def mixup_data(x, y, alpha=1.0, use_cuda=True, concat_ori=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    if not concat_ori or alpha == 0:
        return mixed_x, y_a, y_b, lam
    else:
        cat_x = torch.cat([x, mixed_x])
        cat_y_a = torch.cat([y, y_a])
        cat_y_b = torch.cat([y, y_b])
        return cat_x, cat_y_a, cat_y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
