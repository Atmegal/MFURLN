import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
from lib.model.utils.config import cfg
import cv2
import math
import pdb
import random


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)


def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def save_checkpoint(state, filename):
    torch.save(state, filename)


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

def sigmoid_cross_entropy_loss(pred_scores, pred_labels, dete_labels, gamma=0.5):
    # pred_labels one_hot vector
    pred_labels = pred_labels.float()
    dete_labels = dete_labels.float()
    pred_probs = torch.sigmoid(pred_scores).clamp(1e-8, 1.0)
    pred_logs_fg = torch.log(pred_probs)
    pred_logs_bg = torch.log(1.0 - pred_probs)
    pred_labels_bg = 1.0 - pred_labels
    sig_loss_bg = torch.sum(pred_labels_bg * pred_logs_bg, 1)
    sig_cross_loss = -torch.sum(pred_labels * pred_logs_fg) - \
                     torch.sum(sig_loss_bg.view(-1,1) * dete_labels) * gamma

    return torch.mean(sig_cross_loss)


def dete_cross_loss(dete_scores, dete_labels):
    # dete_labels : 1 determinate 0 indetermiate
    dete_labels = dete_labels.float()
    dete_probs = torch.sigmoid(dete_scores).clamp(1e-8, 1.0)
    dete_logs_fg = torch.log(dete_probs)
    dete_logs_bg = torch.log(1.0 - dete_probs)
    dete_labels_bg = 1.0 - dete_labels
    dete_loss = -torch.sum(dete_labels * dete_logs_fg) - torch.sum(dete_labels_bg * dete_logs_bg)

    return torch.mean(dete_loss)


def _crop_pool_layer(bottom, rois, max_pool=True):
    # code modified from
    # https://github.com/ruotianluo/pytorch-faster-rcnn
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()
    batch_size = bottom.size(0)
    D = bottom.size(1)
    H = bottom.size(2)
    W = bottom.size(3)
    roi_per_batch = rois.size(0) / batch_size
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = bottom.size(2)
    width = bottom.size(3)

    # affine theta
    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([ \
        (x2 - x1) / (width - 1),
        zero,
        (x1 + x2 - width + 1) / (width - 1),
        zero,
        (y2 - y1) / (height - 1),
        (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    if max_pool:
        pre_pool_size = cfg.POOLING_SIZE * 2
        grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))
        bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W) \
            .contiguous().view(-1, D, H, W)
        crops = F.grid_sample(bottom, grid)
        crops = F.max_pool2d(crops, 2, 2)
    else:
        grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE)))
        bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W) \
            .contiguous().view(-1, D, H, W)
        crops = F.grid_sample(bottom, grid)

    return crops, grid


def _affine_grid_gen(rois, input_size, grid_size):
    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([ \
        (x2 - x1) / (width - 1),
        zero,
        (x1 + x2 - width + 1) / (width - 1),
        zero,
        (y2 - y1) / (height - 1),
        (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid


def _affine_theta(rois, input_size):
    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())

    # theta = torch.cat([\
    #   (x2 - x1) / (width - 1),
    #   zero,
    #   (x1 + x2 - width + 1) / (width - 1),
    #   zero,
    #   (y2 - y1) / (height - 1),
    #   (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    theta = torch.cat([ \
        (y2 - y1) / (height - 1),
        zero,
        (y1 + y2 - height + 1) / (height - 1),
        zero,
        (x2 - x1) / (width - 1),
        (x1 + x2 - width + 1) / (width - 1)], 1).view(-1, 2, 3)

    return theta




class FC(nn.Module):
    def __init__(self, in_size, out_size, drop_rate=0.):
        super(FC, self).__init__()
        self.drop_rate = drop_rate
        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU(inplace=True)
        if self.drop_rate > 0.:
            self.dropout = nn.Dropout(self.drop_rate)
        weights_normal_init(self.linear, dev=0.001)

    def forward(self, x):
        x = self.relu(self.linear(x))
        if self.drop_rate > 0.:
            x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']


class NewNoamOpt(object):
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, warmup, decay_start, decay_gamma, max_lr, datasize, batchsize):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.max_lr = max_lr
        self.datasize = datasize
        self.batchsize = batchsize
        self.decay_start = decay_start
        self.decay_gamma = decay_gamma
        self.iterforepoch = math.ceil(self.datasize / self.batchsize)

    def step(self, epoch):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate(epoch)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, epoch):
        "Implement `lrate` above"
        step = self._step
        if epoch < self.warmup:
            return step * self.max_lr / ((self.warmup) * self.iterforepoch)
        elif epoch >= self.warmup and epoch < self.decay_start + 1 :
            return get_lr(self.optimizer)
        else:
            return max(self.max_lr * (self.decay_gamma ** int( (epoch-1) / (self.decay_start))), 1.0e-6)


    def __getattr__(self, name):
        return getattr(self.optimizer, name)
