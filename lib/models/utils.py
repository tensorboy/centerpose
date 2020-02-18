from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

def _gather_feat(feat, ind, mask=None):
    """
    params:
        feat:[batch,height*width,channel]
        ind :[batch, maxobject]
    """
    #dim:channel
    dim  = feat.size(2)
    #ind: [batch, maxobject, channel]
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    #feat: [batch, maxobject, channel]
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    """
    params:
        feat:[batch,channel,height,width]
        ind :[batch, maxobject]
    """
    #[batch,channel,height,width]->[batch,height,width,channel]
    feat = feat.permute(0, 2, 3, 1).contiguous()
    #[batch,height,width,channel]->[bacth,height*width,channel]
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])

def flip_lr(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
            tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
    tmp[:, :, 0, :, :] *= -1
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
            tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)
