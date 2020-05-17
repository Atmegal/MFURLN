from __future__ import absolute_import
import torch
import numpy as np
import torch.nn as nn
from .bbox_transform import bbox_overlaps
import pdb

class _Proposal_rela_layer(nn.Module):

    def __init__(self, num_classes):
        super(_Proposal_rela_layer, self).__init__()
        self.n_classes = num_classes
        #self.n_preds = num_predicates

    def forward(self, rois, cls_pred, gt_boxes, sub_boxes, obj_boxes, rela_gt, num_box):

        gt_boxes = gt_boxes[:num_box.long()]
        gt_rois = gt_boxes.new(gt_boxes.shape[0], 5).zero_()
        gt_rois[:, 1:5] = gt_boxes[:, :4]
        gt_cls = gt_boxes[:, 4]
        rois = torch.cat((rois, gt_rois), 0)
        cls_pred = torch.cat((cls_pred, gt_cls), 0)

        A = rois.shape[0]
        matrix = torch.ones(A, A) - torch.eye(A)
        index = torch.nonzero( matrix > 0).view(-1, 2)
        sub_inds = index[:, 0]
        obj_inds = index[:, 1]

        rela_num = torch.sum((sub_boxes[:, 4]!=0))
        sub_boxes = sub_boxes[:rela_num]
        obj_boxes = obj_boxes[:rela_num]
        rela_gt = rela_gt[:rela_num]

        so_gt = torch.cat((sub_boxes[:, 4], obj_boxes[:, 4]), 0).view(1, -1)
        overlaps = bbox_overlaps(rois[:, 1:5], torch.cat([sub_boxes[:, :4], obj_boxes[:,:4]], 0))
        overlaps = torch.where(overlaps >= 0.5, overlaps, torch.zeros_like(overlaps))

        cls_pred = cls_pred.view(-1, 1)
        pred_matrix = ((cls_pred - so_gt) == 0).float()
        sub_pred = pred_matrix[:, :sub_boxes.shape[0]]
        obj_pred = pred_matrix[:, sub_boxes.shape[0]:]

        sub_pred_score = overlaps[:, :sub_boxes.shape[0]] * sub_pred
        obj_pred_score = overlaps[:, sub_boxes.shape[0]:] * obj_pred

        sub_det = sub_pred_score[sub_inds]
        obj_det = obj_pred_score[obj_inds]
        pairs_scores = sub_det * obj_det
        scores = torch.sum(pairs_scores, dim=1)

        fg_inds = torch.nonzero(scores > 0).view(-1)
        bg_inds = torch.nonzero(scores == 0).view(-1)

        fg_per_images = 32
        rois_per_images = 128
        num_fg = min(fg_per_images, fg_inds.numel())
        num_bg = min((rois_per_images - num_fg), bg_inds.numel())
        rand_num_fg = torch.from_numpy(np.random.permutation(fg_inds.numel())).type_as(gt_boxes).long()
        rand_num_bg = torch.from_numpy(np.random.permutation(bg_inds.numel())).type_as(gt_boxes).long()

        fg_inds = fg_inds[rand_num_fg[: num_fg]]
        bg_inds = bg_inds[rand_num_bg[: num_bg]]
        keeps = torch.cat([fg_inds, bg_inds], 0)
        rela_inds = pairs_scores[keeps].argmax(1)
        rela_labels = rela_gt[rela_inds]
        rela_labels[fg_inds.size(0):] = 0

        sub_indexs = sub_inds[keeps]
        obj_indexs = obj_inds[keeps]
        sub_labels = cls_pred[sub_indexs].long()
        obj_labels = cls_pred[obj_indexs].long()

        return rela_labels, rois, cls_pred, sub_labels, sub_indexs, obj_labels, obj_indexs