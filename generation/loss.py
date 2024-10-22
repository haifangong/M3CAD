import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import lambertw


def kl_gaussianprior(mu, logvar):
    """ analytically compute kl divergence with unit gaussian. """
    var_reg = torch.mean(0.5 * torch.sum((logvar.exp() - 1 - logvar), 1))
    return torch.mean(0.5 * torch.sum((logvar.exp() + mu ** 2 - 1 - logvar), 1))


def compute_ce_loss_part(seq_rec, seq_gt):
    """
    Computes the CrossEntropyLoss between the decoder's output and the target sequences.

    Args:
        seq_rec (torch.Tensor): Decoder output logits of shape (batch_size, sequence_length, num_classes)
        seq (torch.Tensor): Target sequences of shape (batch_size, 1, sequence_length)
        num_classes (int): Number of classes (default is 21 for 20 amino acids + 1 padding)

    Returns:
        torch.Tensor: Scalar loss value
    """
    # Remove the singleton second dimension: (batch_size, 1, sequence_length) -> (batch_size, sequence_length)
    target = seq_gt.squeeze(1)  # Shape: (batch_size, sequence_length)

    # Permute seq_rec to (batch_size, num_classes, sequence_length)
    # because CrossEntropyLoss expects (batch, num_classes, ...)
    seq_rec = seq_rec.permute(0, 2, 1)  # Shape: (batch_size, num_classes, sequence_length)
    loss_rec_seq = 0
    for i in range(seq_rec.shape[0]):
        gt_item = target[i]
        rec_item = seq_rec[i]
        zero_indices = torch.where(gt_item == 0)[0][0] + 1

        clip_gt = gt_item[:zero_indices].unsqueeze(0)
        # clip_gt_v2 = F.one_hot(clip_gt.long(), num_classes=21)
        clip_rec = rec_item[..., :zero_indices].unsqueeze(0).float()
        loss_rec_seq += F.cross_entropy(clip_rec, clip_gt.long())
        # print()
        # loss_rec_seq += F.mse_loss(clip_gt_v2.float(), clip_rec)
        # clip_rec_v2 = clip_rec.clone()
        # print(clip_gt.shape)
        # print(clip_rec.shape)
        # loss_rec_seq += F.cross_entropy(torch.cat((clip_gt, clip_gt_v2), dim=0), torch.cat((clip_rec, clip_rec_v2), dim=0))

    return loss_rec_seq / seq_rec.shape[0]


def compute_ce_loss(seq_rec, seq_gt):
    """
    Computes the CrossEntropyLoss between the decoder's output and the target sequences.

    Args:
        seq_rec (torch.Tensor): Decoder output logits of shape (batch_size, sequence_length, num_classes)
        seq (torch.Tensor): Target sequences of shape (batch_size, 1, sequence_length)
        num_classes (int): Number of classes (default is 21 for 20 amino acids + 1 padding)

    Returns:
        torch.Tensor: Scalar loss value
    """
    # Remove the singleton second dimension: (batch_size, 1, sequence_length) -> (batch_size, sequence_length)
    target = seq_gt.squeeze(1)  # Shape: (batch_size, sequence_length)

    # Ensure target is of type LongTensor
    target = target.long()

    # Permute seq_rec to (batch_size, num_classes, sequence_length)
    # because CrossEntropyLoss expects (batch, num_classes, ...)
    seq_rec = seq_rec.permute(0, 2, 1)  # Shape: (batch_size, num_classes, sequence_length)

    # Compute the loss
    loss_rec_seq = F.cross_entropy(seq_rec, target, ignore_index=21, reduction='mean')

    return loss_rec_seq


def compute_mmd(x, y, kernel='rbf', sigma=1.0):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    if kernel == 'rbf':
        xx = x.unsqueeze(1).expand(x_size, y_size, dim)
        yy = y.unsqueeze(0).expand(x_size, y_size, dim)
        L2_distance = ((xx - yy) ** 2).sum(2)
        # Compute kernel values
        K = torch.exp(-L2_distance / (2 * sigma ** 2))
        mmd = K.mean()
        return mmd
    else:
        raise NotImplementedError('Only RBF kernel is implemented.')


# def vae_loss(recon_x, x, mean, log_var):
#     # BCE = torch.nn.functional.binary_cross_entropy(
#         # recon_x, x, reduction='mean')
#     BCE = torch.nn.functional.smooth_l1_loss(recon_x, x)
#     KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#
#     return (BCE + KLD) / x.size(0)

# def onehot_embed(hardIx, vocabSize):
#     """ Get tensor hardIx (mbsize), return it's one hot embedding  (mbsize x vocabSize) """
#     assert hardIx.dim() == 1, 'expecting 1D tensor: minibatch of indices.'
#     softIx = torch.zeros(hardIx.size(0), vocabSize).to(hardIx.device)
#     softIx.scatter_(1, hardIx.unsqueeze(1), 1.0)
#     return softIx
# seq = torch.ones((16, 20))
# one_hot_list = []
# for i in range(seq.shape[0]):
#     one_hot_list.append(F.one_hot(seq[i, 0, :].to(torch.int64), num_classes=22).unsqueeze(0))
# one_hot_seq = torch.cat((one_hot_list), dim=0).float()
def recon_dec(sequences, logits):
    """ compute reconstruction error (NLL of next-timestep predictions) """
    # dec_inputs: '<start> I want to fly <eos>'
    # dec_targets: 'I want to fly <eos> <pad>'
    # sequences: [mbsize x seq_len]
    # logits: [mbsize x seq_len x vocabsize]
    mbsize = sequences.size(0)
    pad_words = torch.LongTensor(mbsize, 1).fill_(-1).to(sequences.device)
    dec_targets = torch.cat([sequences[:, 1:], pad_words], dim=1)
    recon_loss = F.cross_entropy(  # this is log_softmax + nll
        logits.view(-1, logits.size(2)), dec_targets.view(-1), reduction='mean',
        ignore_index=-1  # padding doesnt contribute to recon loss & gradient
    )
    return recon_loss


def vae_loss(recon_x, x, mean, log_var):
    # BCE = torch.nn.functional.binary_cross_entropy(
    # recon_x, x, reduction='mean')
    vox_rec, seq_rec = recon_x
    vox, seq = x
    # BCE = torch.nn.functional.smooth_l1_loss(vox_rec, vox)
    BCE = 0

    # print(seq_rec.shape)
    # print(seq.shape)
    seq_loss = 0
    seq = seq.squeeze(1)
    for i in range(seq_rec.shape[0]):
        length = torch.where(seq[i] >= 0)[-1][-1].item() + 1
        seq_loss += torch.nn.functional.l1_loss(seq_rec[i][:length], seq[i][:length])
    seq_loss /= seq_rec.shape[0]
    # print('rec_seq_loss', seq_loss)
    # print('rec_vox_loss', BCE)
    SEQ = seq_loss
    # SEQ = torch.nn.functional.l1_loss(seq_rec, seq)
    # print(seq_rec[0])
    # print(seq[0])
    # SEQ = extended_edit_distance(seq_rec, seq)
    # SEQ = F.cross_entropy(  # this is log_softmax + nll
    #     seq_rec.flatten(), seq.flatten(), reduction='mean',
    #     ignore_index=-1  # padding doesnt contribute to recon loss & gradient
    # )
    KLD = 0

    return (BCE + SEQ + KLD) / x[0].size(0)


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
