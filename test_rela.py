# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
from lib.roi_data_layer.roibatchLoader_rela import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.faster_rcnn.vgg16 import vgg16
from lib.model.faster_rcnn.resnet import resnet
from lib.model.utils.funcs import all_recall
from lib.datasets.evaluator import evaluation
from random import shuffle

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
                        help='directory to load models', default="outputs",
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
                        default=6, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=6999, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args


def test_epoch(model, valloader, num_images, num_classes):
    model.eval()
    valdata = iter(valloader)
    P_R50 = 0.0
    R_R50 = 0.0
    P_R100 = 0.0
    R_R100 = 0.0
    N_total = 0.0
    dete_boxes = [[] for _ in range(num_classes)]
    test_gt_boxes = [[[] for _ in range(num_classes)] for _ in range(num_images)]

    for i in range(num_images):
        data = next(valdata)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])
            sub_boxes.resize_(data[4].size()).copy_(data[4])
            obj_boxes.resize_(data[5].size()).copy_(data[5])
            rela_gt.resize_(data[6].size()).copy_(data[6])
            det_tic = time.time()

            rela_cls_pred, rela_pred_score, sub_pred_boxes, sub_pred_cls, sub_pred_scores, \
            obj_pred_boxes, obj_pred_cls, obj_pred_scores, all_pred_boxes, all_pred_cls, all_pred_scores \
                = model(im_data, im_info, gt_boxes, num_boxes, sub_boxes, obj_boxes, rela_gt)


            num_rela = (sub_boxes[0][:, 4] > 0).sum().int()
            roidb_gt = {'rela_gt': rela_gt[0][:num_rela].cpu().numpy(),
                        'sub_box_gt': sub_boxes[0][:num_rela, :4].cpu().numpy(),
                        'sub_gt': sub_boxes[0][:num_rela, 4].cpu().numpy(),
                        'obj_box_gt': obj_boxes[0][:num_rela, :4].cpu().numpy(),
                        'obj_gt': obj_boxes[0][:num_rela, 4].cpu().numpy()}

            rela_scores = rela_pred_score *  sub_pred_scores * obj_pred_scores
            roidb_pred = {'pred_rela': rela_cls_pred.cpu().numpy(),
                          'pred_rela_score': rela_scores.cpu().numpy(),
                          'sub_box_dete': sub_pred_boxes.cpu().numpy(),
                          'sub_dete': sub_pred_cls.cpu().numpy(),
                          'obj_box_dete': obj_pred_boxes.cpu().numpy(),
                          'obj_dete': obj_pred_cls.cpu().numpy()}

            rela_r50, phr_r50, num = all_recall([roidb_gt], [roidb_pred], 50, 70)
            rela_r100, phr_r100, _ = all_recall([roidb_gt], [roidb_pred], 100, 70)

            P_R50 = P_R50 + phr_r50
            P_R100 = P_R100 + phr_r100
            R_R50 = R_R50 + rela_r50
            R_R100 = R_R100 + rela_r100
            N_total = N_total + num

            misc_toc = time.time()
            total_time = misc_toc - det_tic
            if (i + 1) % 100 == 0:
                print('im_detect: {:d}/{:d} {:.4f}s'.format(i + 1, num_images, total_time))
            # mAP
            gt_box = gt_boxes[0][:num_boxes[0], :4].cpu().numpy()
            gt_class = gt_boxes[0][:num_boxes[0], 4:5].cpu().numpy()
            # pdb.set_trace()
            for j, c in enumerate(gt_class):
                test_gt_boxes[i][int(c)].append(gt_box[j])
            for k, c in enumerate(all_pred_cls.cpu().numpy()):
                pred_temp = {'image_id': i,
                             'bbox': all_pred_boxes[k].cpu().numpy(),
                             'scores': all_pred_scores[k].cpu().numpy()}
                dete_boxes[int(c)].append(pred_temp)

    aps = []
    for j in range(1, num_classes):
        gt_roidb = [x[j] for x in test_gt_boxes]
        dete_roidb = dete_boxes[j]
        ap = evaluation(gt_roidb, dete_roidb, j, iou_thresh=0.5, use_07_metric=False)
        aps.append(ap)

    mAP = np.mean(aps)
    P_R50 = P_R50 / N_total
    R_R50 = R_R50 / N_total
    P_R100 = P_R100 / N_total
    R_R100 = R_R100 / N_total
    print('mAP: {:4f}, phrase_R50: {:.4f}, phrase_R100: {:.4f}, rela_R50: {:.4f}, rela_R100: {:.4f}'.
          format( mAP, P_R50, P_R100, R_R50, R_R100))
    f = open('./results/test.txt', "a")
    f.write('vrd')
    f.write('   ')
    f.write('mAP: ')
    f.write('%.4f' % (mAP))
    f.write('   ')
    f.write('P_R@50: ')
    f.write('%.4f' % (P_R50))
    f.write('   ')
    f.write('P_R@100: ')
    f.write('%.4f' % (P_R100))
    f.write('   ')
    f.write('R_R@50: ')
    f.write('%.4f' % (R_R50))
    f.write('   ')
    f.write('R_R@100: ')
    f.write('%.4f' % (R_R100))
    f.write('\n')
    f.close()


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    args = parse_args()
    args.cuda = True
    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "VRD":
        args.imdb_name = "./process/vrd_rela_roidb.npz"
        # args.imdb_name = "./process/vrd_zero_shot.npz"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        num_class = 101
        num_predicate = 71
    elif args.dataset == "VG":
        args.imdb_name = "./process/vg_rela_roidb.npz"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        num_class = 151
        num_predicate = 51

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
    # roidb = roidb_temp['test_zero_shot']
    test_size = len(roidb)
    print('{:d} roidb entries'.format(test_size))

    input_dir = args.load_dir + "/relations/vrd/mfurln/"
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,'faster_rcnn_{}_{}_{}.pth'.format(
        args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(num_class, num_predicate, pretrained=False,
                           class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(num_class, 50, num_predicate, pretrained=True,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(num_class, 101, num_predicate, pretrained=True,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(num_class, 152, num_predicate, pretrained=True,
                            class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    model_dict = fasterRCNN.state_dict()
    state_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    fasterRCNN.load_state_dict(model_dict)

    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    print('load model successfully!')

    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    im_scale = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    obj_boxes = torch.FloatTensor(1)
    sub_boxes = torch.FloatTensor(1)
    rela_gt = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        im_scale = im_scale.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        obj_boxes = obj_boxes.cuda()
        sub_boxes = sub_boxes.cuda()
        rela_gt = rela_gt.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    im_scale = Variable(im_scale)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    obj_boxes = Variable(obj_boxes)
    sub_boxes = Variable(sub_boxes)
    rela_gt = Variable(rela_gt)

    if args.cuda:
        cfg.CUDA = True
    if args.cuda:
        fasterRCNN.cuda()

    dataset = roibatchLoader(roidb, 1, num_class, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=4, pin_memory=True)
    val_epoch(fasterRCNN, dataloader, test_size, num_class)