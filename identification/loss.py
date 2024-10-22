import torch
from torch import nn
import torch.nn.functional as F

import math
import numpy as np
from scipy.special import lambertw


def mixup_criterion(criterion, pred, y_a, y_b, lam, pow=2):
    y = lam ** pow * y_a + (1 - lam) ** pow * y_b
    return criterion(pred, y)


def mixup_data(v, q, a):
    '''Returns mixed inputs, pairs of targets, and lambda without organ constraint'''
    lam = np.random.beta(1, 1)

    batch_size = v.shape[0]
    index = torch.randperm(batch_size)

    mixed_v = lam * v + (1 - lam) * v[index, :]
    mixed_q = lam * q + (1 - lam) * q[index, :]

    a_1, a_2 = a, a[index]
    return mixed_v, mixed_q, a_1, a_2, lam


class MLCE(nn.Module):
    def __init__(self):
        super(MLCE, self).__init__()

    def _mlcce(self, y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        loss = torch.mean(neg_loss + pos_loss)
        return loss

    def __call__(self, y_pred, y_true):
        return self._mlcce(y_pred, y_true)


class SuperLoss(nn.Module):
    def __init__(self, C=10, lam=1, batch_size=1):
        super(SuperLoss, self).__init__()
        self.tau = math.log(C)
        self.lam = lam  # set to 1 for CIFAR10 and 0.25 for CIFAR100
        self.batch_size = batch_size

    def forward(self, logits, targets):
        l_i = F.mse_loss(logits, targets, reduction='none').detach()
        sigma = self.sigma(l_i)
        loss = (F.mse_loss(logits, targets, reduction='none') - self.tau) * sigma + self.lam * (
                    torch.log(sigma) ** 2)
        loss = loss.sum() / self.batch_size
        return loss

    def sigma(self, l_i):
        x = torch.ones(l_i.size()) * (-2 / math.exp(1.))
        x = x.cuda()
        y = 0.5 * torch.max(x, (l_i - self.tau) / self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()
        return sigma

