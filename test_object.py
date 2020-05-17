# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import pprint
import pdb
import time

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.nms.nms_wrapper import nms
from lib.model.faster_rcnn.vgg16 import vgg16
from lib.model.faster_rcnn.resnet import resnet
from lib.datasets.evaluator import evaluation
from random import shuffle
try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='VRD', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='vgg16', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size', default=1,
                         type=int)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="outputs/objects",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=8, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=583, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args


def test_epoch(model, valloader, num_images, num_classes, im_data,
               im_info, num_boxes, gt_boxes):
    model.eval()
    valdata = iter(valloader)
    dete_boxes = [[] for _ in range(num_classes)]
    test_gt_boxes = [[[] for _ in range(num_classes)] for _ in range(num_images)]
    i = 0
    while (i < num_images):
        data = next(valdata)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])
            det_tic = time.time()

            all_pred_boxes, all_pred_cls, all_pred_scores = model(im_data, im_info, gt_boxes, num_boxes)

            # b is batchsize
            for b in range(all_pred_cls.size(0)):
                if i >= num_images:
                    break
                misc_toc = time.time()
                total_time = misc_toc - det_tic
                if (i + 1) % 100 == 0:
                    print('im_detect: {:d}/{:d} {:.4f}s'.format(i + 1, num_images, total_time))

                # mAP
                gt_box = gt_boxes[b][:num_boxes[b], :4].cpu().numpy()
                gt_class = gt_boxes[b][:num_boxes[b], 4:5].cpu().numpy()
                for j, c in enumerate(gt_class):
                    test_gt_boxes[i][int(c)].append(gt_box[j])
                for k, c in enumerate(all_pred_cls[b].cpu().numpy()):
                    pred_temp = {'image_id': i,
                                 'bbox': all_pred_boxes[b][k].cpu().numpy(),
                                 'scores': all_pred_scores[b][k].cpu().numpy()}
                    dete_boxes[int(c)].append(pred_temp)
                i = i + 1

    aps = []
    for i in range(1, num_classes):
        gt_roidb = [x[i] for x in test_gt_boxes]
        dete_roidb = dete_boxes[i]
        ap = evaluation(gt_roidb, dete_roidb, i, iou_thresh=0.5, use_07_metric=False)
        aps.append(ap)
    mAP = np.mean(aps)
    print('mAP: {:4f}'.format(mAP))
    f = open('./results/vrd_object.txt', "a")
    f.write('vrd test')
    f.write('    ')
    f.write('mAP: ')
    f.write('%.4f' % (mAP))
    f.write('\n')
    f.close()


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "VRD":
        args.imdb_name = "./process/vrd_object_roidb.npz"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        num_class = 101

    elif args.dataset == "VG":
        args.imdb_name = "./process/vg_object_roidb.npz"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        num_class = 151

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    load_roidb = np.load(args.imdb_name, allow_pickle=True)
    roidb_temp = load_roidb['roidb'][()]
    roidb = roidb_temp['test_roidb']
    data_size = len(roidb)
    if data_size % args.batch_size:
        choose_size = np.array(np.arange(data_size))
        shuffle(choose_size)
        size = args.batch_size - data_size%args.batch_size
        for i in np.arange(size):
            roidb.append(roidb_temp['test_roidb'][choose_size[i]])

    print('{:d} roidb entries'.format(len(roidb)))
    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(num_class, pretrained=False,
                           class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(num_class, 50, pretrained=False,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(num_class, 101, pretrained=True,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(num_class, 152, pretrained=True,
                            class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    model_dict = fasterRCNN.state_dict()
    state_dict = {k: v for k, v in checkpoint['model'].items() if  k in model_dict.keys()}
    model_dict.update(state_dict)
    fasterRCNN.load_state_dict(model_dict)
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    print('load model successfully!')

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    im_scale = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        im_scale = im_scale.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    im_scale = Variable(im_scale)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True
    if args.cuda:
        fasterRCNN.cuda()

    dataset = roibatchLoader(roidb, args.batch_size, num_class, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle = False, num_workers=8,
                                             pin_memory=True)

    test_epoch(fasterRCNN, dataloader, data_size, num_class, im_data, im_info, num_boxes, gt_boxes)
