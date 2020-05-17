"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from lib.model.utils.config import cfg
from lib.roi_data_layer.minibatch_rela import get_minibatch
import numpy as np

class roibatchLoader(data.Dataset):
    def __init__(self, roidb, batch_size, num_classes, training=True, normalize=None):
        self._roidb = roidb
        self._num_classes = num_classes
        # we make the height of image consistent to trim_height, trim_width
        self.trim_height = cfg.TRAIN.TRIM_HEIGHT
        self.trim_width = cfg.TRAIN.TRIM_WIDTH
        self.max_num_box = cfg.MAX_NUM_GT_BOXES
        self.training = training
        self.normalize = normalize
        self.batch_size = batch_size
        self.data_size = len(self._roidb)

    def __getitem__(self, index):

        minibatch_db = [self._roidb[index]]
        blobs = get_minibatch(minibatch_db, self._num_classes, self.training)
        data = torch.from_numpy(blobs['data'])
        im_info = torch.from_numpy(blobs['im_info'])
        # we need to random shuffle the bounding box.
        data_height, data_width = data.size(1), data.size(2)

        # np.random.shuffle(blobs['gt_boxes'])
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])
        sub_boxes = torch.from_numpy(blobs['sub_gt_boxes'])
        rela_gt = torch.from_numpy(blobs['rela_gt'])
        obj_boxes = torch.from_numpy(blobs['obj_gt_boxes'])
        ########################################################
        # padding the input image to fixed size for each group #
        ########################################################
        # cfg.TRAIN.MAX_SIZE == 800
        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        padding_data = torch.FloatTensor(3, cfg.TRAIN.MAX_SIZE, cfg.TRAIN.MAX_SIZE).zero_()
        if data_height > data_width:
            padding_data[:, :, :data_width] = data
        else:
            padding_data[:, :data_height, :] = data

        gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
        num_boxes = min(gt_boxes.size(0), self.max_num_box)
        gt_boxes_padding[:num_boxes, :] = gt_boxes[:num_boxes]

        sub_boxes_padding = torch.FloatTensor(self.max_num_box, sub_boxes.size(1)).zero_()
        sub_num_boxes = min(sub_boxes.size(0), self.max_num_box)
        sub_boxes_padding[:sub_num_boxes, :] = sub_boxes[:sub_num_boxes]

        obj_boxes_padding = torch.FloatTensor(self.max_num_box, obj_boxes.size(1)).zero_()
        obj_num_boxes = min(obj_boxes.size(0), self.max_num_box)
        obj_boxes_padding[:obj_num_boxes, :] = obj_boxes[:obj_num_boxes]

        rela_gt_padding = torch.FloatTensor(self.max_num_box, rela_gt.size(1)).zero_()
        rela_gtnum_boxes = min(rela_gt.size(0), self.max_num_box)
        rela_gt_padding[:rela_gtnum_boxes, :] = rela_gt[:rela_gtnum_boxes]

        im_info = im_info.view(3)
        id = torch.from_numpy(blobs['img_id'])
        return padding_data, im_info, gt_boxes_padding, num_boxes,\
               sub_boxes_padding, obj_boxes_padding, rela_gt_padding, id

    def __len__(self):
        return len(self._roidb)