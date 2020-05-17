import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

# from model.roi_layers import nms
from torchvision.ops import nms
from lib.model.utils.config import cfg
from lib.model.rpn.rpn import _RPN
from lib.model.roi_layers import ROIAlign, ROIPool
from lib.model.rpn.bbox_transform import clip_boxes, bbox_transform_inv
from lib.model.rpn.proposal_target_layer import _ProposalTargetLayer
from lib.model.rpn.proposal_rela_layer import _Proposal_rela_layer
from lib.model.rpn.rela_feature_fusion import _Extract_rela_feature
from lib.model.utils.net_utils import dete_cross_loss, sigmoid_cross_entropy_loss, FC
import time
import pdb

class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, num_classes, num_predicates, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.n_classes = num_classes
        self.class_agnostic = class_agnostic
        self.n_preds = num_predicates
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0)
        self.Proposal_rela_layer = _Proposal_rela_layer(self.n_classes)
        self.Extract_rela_featrue = _Extract_rela_feature(self.n_classes, 4096, 512)

        self.linear_u_rela = FC(4096, 512)
        self.linear_v_rela = FC(1536, 512)
        self.linear_s_rela = FC(512, 512)
        self.linear_l_rela = FC(1536, 512)

        self.linear_f_rela = FC(1536, 100)
        self.linear_d_rela = nn.Linear(100, 1)
        self.linear_r_rela = FC(101, 512)
        # rela scores predict
        self.rela_cls_pred = nn.Linear(512, self.n_preds)

    def forward(self, im_data, im_info, gt_boxes=None, num_boxes=None,
                sub_boxes=None, obj_boxes=None, rela_gt=None):

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        sub_boxes = sub_boxes.data
        obj_boxes = obj_boxes.data
        rela_gt = rela_gt.data

        base_feat = self.RCNN_base(im_data)
        rois, _, _ = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        rois = rois[:, :300].contiguous().view(-1, 5)

        _, bbox_pred, cls_score = self._obj_feature(base_feat, rois)
        cls_prob = F.softmax(cls_score, 1)
        rois = self._rois_bbox(rois, bbox_pred, self.n_classes, im_info)
        obj_rois, obj_cls, obj_scores = self._select_objs(rois, cls_prob, self.n_classes)

        rela_labels = 0.0
        if self.training:
            rela_labels, obj_rois, obj_cls, sub_labels, sub_indexs, obj_labels, obj_indexs = \
                self.Proposal_rela_layer(obj_rois, obj_cls, gt_boxes[0], sub_boxes[0],
                                         obj_boxes[0], rela_gt[0], num_boxes[0])
        else:
            A = obj_rois.shape[0]
            matrix = torch.ones(A, A) - torch.eye(A)
            index = torch.nonzero(matrix > 0).view(-1, 2)
            sub_indexs = index[:, 0]
            obj_indexs = index[:, 1]
            sub_labels = obj_cls.view(-1, 1)[sub_indexs].long()
            obj_labels = obj_cls.view(-1, 1)[obj_indexs].long()
            obj_rois = obj_rois.view(-1, 5)
            obj_scores = obj_scores.view(-1, 1)

        rois_sub = obj_rois[sub_indexs]
        rois_obj = obj_rois[obj_indexs]
        rois_union = self._extract_union_bbox(rois_sub, rois_obj)

        pool5 = 0.
        pool5_union = 0.
        if cfg.POOLING_MODE == 'align':
            pool5 = self.RCNN_roi_align(base_feat, obj_rois)
            pool5_union = self.RCNN_roi_align(base_feat, rois_union)
        elif cfg.POOLING_MODE == 'pool':
            pool5 = self.RCNN_roi_pool(base_feat, obj_rois)
            pool5_union = self.RCNN_roi_pool(base_feat, rois_union)

        obj_fc7 = self._head_to_tail(pool5)
        union_fc7 = self._head_to_tail(pool5_union)
        union_fc = self.linear_u_rela(union_fc7)

        sub_fc, obj_fc, spatial_features, exter_sub, exter_obj, inter_so = \
            self.Extract_rela_featrue(obj_fc7, obj_rois, sub_labels, sub_indexs, obj_labels, obj_indexs)
        # 1536 ---> 512
        feature_v = self.linear_v_rela(torch.cat([sub_fc, obj_fc, union_fc], 1))
        feature_s = self.linear_s_rela(spatial_features)
        feature_l = self.linear_l_rela(torch.cat([exter_sub, exter_obj, inter_so], 1))
        # feature_fusions [-1, 100]
        feature_fusions = self.linear_f_rela(torch.cat([feature_v, feature_s, feature_l], 1))
        # dete_prob 100---> 1
        dete_score = self.linear_d_rela(feature_fusions)
        # 101 ---> 512
        final_feats = self.linear_r_rela(torch.cat((feature_fusions, dete_score), -1))
        # 512 ---> 70
        rela_cls_score = self.rela_cls_pred(final_feats)

        if self.training:
            dete_labels = (torch.sum(rela_labels, 1) > 0).view(-1, 1)
            dete_loss = dete_cross_loss(dete_score, dete_labels)
            rela_loss = sigmoid_cross_entropy_loss(rela_cls_score, rela_labels[:, 1:], dete_labels)

            return dete_loss, rela_loss
        else:

            rela_cls_prob = torch.sigmoid(rela_cls_score)
            rela_pred_score, rela_cls_pred = torch.max(rela_cls_prob, 1)
            dete_prob = torch.sigmoid(dete_score)
            rela_pred_score = rela_pred_score.view(-1, 1) * dete_prob
            obj_boxes = obj_rois[:, 1:].contiguous().view(-1, 4)
            sub_pred_boxes = obj_boxes[sub_indexs]
            obj_pred_boxes = obj_boxes[obj_indexs]
            sub_pred_scores = obj_scores[sub_indexs]
            obj_pred_scores = obj_scores[obj_indexs]

            return rela_cls_pred.view(-1, 1), \
                   rela_pred_score.view(-1, 1), \
                   sub_pred_boxes.view(-1, 4), \
                   sub_labels.view(-1, 1), \
                   sub_pred_scores.view(-1, 1), \
                   obj_pred_boxes.view(-1, 4), \
                   obj_labels.view(-1, 1), \
                   obj_pred_scores.view(-1, 1), \
                   obj_boxes.view(-1, 4), \
                   obj_cls.view(-1, 1), \
                   obj_scores.view(-1, 1)

    def _obj_feature(self, base_feat, rois):
        pooled_feat = 0.
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        cls_score = self.RCNN_cls_score(pooled_feat)

        return pooled_feat, bbox_pred, cls_score

    def create_architecture(self):
        self._init_modules()

    def _rois_bbox(self, rois, bbox, num_classes, im_info):

        rois = rois.repeat(1, num_classes)
        stds = torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
        stds = stds.repeat(1, num_classes)
        means = torch.tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        means = means.repeat(1, num_classes)
        bbox *= stds
        bbox += means
        bbox = bbox.view(-1, 4)
        rois = rois.view(-1, 5)
        rois_label, anchors = torch.split(rois, [1, 4], 1)
        anchors = bbox_transform_inv(anchors.unsqueeze(0), bbox.unsqueeze(0), 1)
        anchors = clip_boxes(anchors, im_info, 1)
        rois = torch.cat([rois_label.view(-1,1), anchors.squeeze(0)], 1).view(-1, 5 * num_classes)

        return rois

    def _select_objs(self, rois, cls_prob, num_classes):

        for i in range(1, num_classes):
            rois_fg = rois[:, 5 * i: 5 * (i + 1)]
            cls_prob_fg = cls_prob[:, i]
            keeps = nms(rois_fg[:, 1:], cls_prob_fg, 0.4).long()
            keeps = keeps[:50]
            if i == 1:
                rois_objs = rois_fg[keeps]
                rois_scores = cls_prob_fg[keeps]
                rois_cls = torch.ones_like(rois_scores) * i
            else:
                rois_objs = torch.cat([rois_objs, rois_fg[keeps]], 0)
                rois_sc = cls_prob_fg[keeps]
                rois_scores = torch.cat([rois_scores, rois_sc])
                rois_cls = torch.cat([rois_cls, torch.ones_like(rois_sc) * i])

        # select top 50
        select_inds = torch.argsort(-rois_scores)
        select_inds = select_inds.view(-1)[:50]
        rois_objs = rois_objs[select_inds]
        rois_scores = rois_scores[select_inds]
        rois_cls = rois_cls[select_inds]

        return rois_objs, rois_cls, rois_scores

    def _extract_union_bbox(self, sub_bbox, obj_bbox):

        union = torch.cat([sub_bbox, obj_bbox], 1)
        u0 = union[:, 0].view(-1, 1)
        u1, _ = torch.min(union[:, 1::5], 1)
        u2, _ = torch.min(union[:, 2::5], 1)
        u3, _ = torch.max(union[:, 3::5], 1)
        u4, _ = torch.max(union[:, 4::5], 1)
        u1 = u1.view(-1, 1)
        u2 = u2.view(-1, 1)
        u3 = u3.view(-1, 1)
        u4 = u4.view(-1, 1)
        union_bbox = torch.cat([u0, u1, u2, u3, u4], 1)

        return union_bbox