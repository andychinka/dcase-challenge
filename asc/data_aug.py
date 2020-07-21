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


def temporal_crop(x, crop_length=400):
    for j in range(x.shape[0]):
        start_loc = np.random.randint(0, x.shape[-1] - crop_length)
        x[j, :, :, 0:crop_length] = x[j, :, :, start_loc:start_loc + crop_length]

    x = x[:, :, :, 0:crop_length]

    return x

def mixup_and_temporal_crop(x, y, alpha=1.0, use_cuda=True, concat_ori=False, crop_length=400):
    # _, h, w, c = self.X_train.shape
    batch_size = x.size()[0]
    lam = np.random.beta(alpha, alpha, batch_size)
    x_lam = lam.reshape(batch_size, 1, 1, 1)
    y_lam = lam.reshape(batch_size, 1)

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    x1 = x
    x2 = x[index, :]

    for j in range(x1.shape[0]):
        start_loc1 = np.random.randint(0, x1.shape[-1] - crop_length)
        start_loc2 = np.random.randint(0, x2.shape[-1] - crop_length)

        x1[j, :, :, 0:crop_length] = x1[j, :, :, start_loc1:start_loc1 + crop_length]
        x2[j, :, :, 0:crop_length] = x2[j, :, :, start_loc2:start_loc2 + crop_length]


    # cropped
    x1 = x1[:, :, :, 0:crop_length]
    x2 = x2[:, :, :, 0:crop_length]

    mixed_x = x1 * x_lam + x2 * (1.0 - x_lam)
    y1, y2 = y, y[index]
    mixed_y = y1 * y_lam + y2 * (1.0 - y_lam)

    if not concat_ori or alpha == 0:
        return mixed_x, mixed_y, lam
    else:
        cat_x = torch.cat([x, mixed_x])
        cat_y = torch.cat([y, mixed_y])
        return cat_x, cat_y, lam



