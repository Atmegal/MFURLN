# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torchvision.models as models
from lib.model.faster_rcnn.mfurln import _fasterRCNN


class vgg16(_fasterRCNN):
    def __init__(self, num_classes, num_predicates, pretrained=False, class_agnostic=False):
        self.model_path = 'data/pretrained_model/vgg16_coco.pth'
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        _fasterRCNN.__init__(self, num_classes, num_predicates, class_agnostic)

    def _init_modules(self):
        vgg = models.vgg16()
        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        # not using the last maxpool layer
        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        self.RCNN_top = vgg.classifier
        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7

