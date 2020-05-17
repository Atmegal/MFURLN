import torch
import torch.nn as nn
import numpy as np
import pdb
from lib.model.utils.net_utils import FC
from torch.autograd import Variable

class _Extract_rela_feature(nn.Module):

    def __init__(self, num_class, in_size=4096, out_size=512):
        super(_Extract_rela_feature, self).__init__()
        self.num_class = num_class
        word2vec = np.load('./process/object_w2v.npy')
        self.external_lan = torch.from_numpy(word2vec).cuda()
        conf = np.load('./process/vrd_internal.npz')
        self.sub_obj_pred = torch.from_numpy(conf['sub_obj']).cuda()
        # visual
        self.linear_vs = FC(in_size, out_size)
        self.linear_vo = FC(in_size, out_size)
        # spatial
        self.linear_sp = FC(8, out_size)
        # language
        self.linear_es = FC(300, out_size)
        self.linear_eo = FC(300, out_size)
        self.linear_in = FC(70, out_size)

    def forward(self, fc7, rois, sub_cls, sub_inds, obj_cls, obj_inds):
        sub_fc = self.linear_vs(fc7)
        obj_fc = self.linear_vo(fc7)
        sub_fc = sub_fc[sub_inds]
        obj_fc = obj_fc[obj_inds]
        sub_bbox = rois[sub_inds]
        obj_bbox = rois[obj_inds]
        spatial_features = self.extract_spatial_feature(sub_bbox, obj_bbox)
        # (sub_labels - 1)  do'not include background
        exter_sub, exter_obj, inter_so= self.extract_language_feature(sub_cls - 1, obj_cls - 1)

        return sub_fc, obj_fc, spatial_features, exter_sub, exter_obj, inter_so

    def extract_spatial_feature(self, sub_bbox, obj_bbox):

        bbox_cat = torch.cat([sub_bbox, obj_bbox], 1)
        u1, _ = torch.min(bbox_cat[:, 1::5], 1)
        u2, _ = torch.min(bbox_cat[:, 2::5], 1)
        u3, _ = torch.max(bbox_cat[:, 3::5], 1)
        u4, _ = torch.max(bbox_cat[:, 4::5], 1)

        u1 = u1.view(-1, 1)
        u2 = u2.view(-1, 1)
        u3 = u3.view(-1, 1)
        u4 = u4.view(-1, 1)

        su1 = (sub_bbox[:, 1:2] - u1) / (u3 - u1)
        su2 = (sub_bbox[:, 2:3] - u2) / (u4 - u2)
        su3 = (sub_bbox[:, 3:4] - u3) / (u3 - u1)
        su4 = (sub_bbox[:, 4:5] - u4) / (u4 - u2)

        ou1 = (obj_bbox[:, 1:2] - u1) / (u3 - u1)
        ou2 = (obj_bbox[:, 2:3] - u2) / (u4 - u2)
        ou3 = (obj_bbox[:, 3:4] - u3) / (u3 - u1)
        ou4 = (obj_bbox[:, 4:5] - u4) / (u4 - u2)

        spatial_feature = torch.cat([su1, su2, su3, su4, ou1, ou2, ou3, ou4], 1)
        spatial_feature = self.linear_sp(spatial_feature)

        return spatial_feature

    def extract_language_feature(self, sub_labels, obj_labels):
        external_sub = self.external_lan[sub_labels.view(-1)]
        external_sub = self.linear_es(external_sub)
        external_obj = self.external_lan[obj_labels.view(-1)]
        external_obj = self.linear_eo(external_obj)
        internal_so = self.sub_obj_pred[sub_labels.view(-1), obj_labels.view(-1)].float()
        internal_so = self.linear_in(internal_so)
        return external_sub, external_obj, internal_so