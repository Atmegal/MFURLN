# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
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
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from random import shuffle
import torchvision.transforms as transforms
from lib.roi_data_layer.roibatchLoader import roibatchLoader, rank_roidb_ratio
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.utils.net_utils import save_checkpoint, NewNoamOpt
from lib.model.faster_rcnn.vgg16 import vgg16
from lib.model.faster_rcnn.resnet import resnet
from lib.datasets.evaluator import evaluation

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='VRD', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=0, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=10, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="outputs/objects",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=8, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=8, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.01, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=4, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)
    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')

    args = parser.parse_args()
    return args


def train_epoch(model, dataloader, optimizer, epoch, iters_per_epoch, output_dir):
        model.train()
        loss_temp = 0
        start = time.time()
        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])

            model.zero_grad()
            rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, rois_label\
                = model(im_data, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(epoch)

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)
                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, optimizer.rate(epoch)))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                       (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()

        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

def val_epoch(model, valloader, epoch, num_images, num_classes, output_dir):
    model.eval()
    valdata = iter(valloader)
    lr = optimizer.rate(epoch)
    dete_boxes = [[] for _ in range(num_classes)]
    val_gt = [[[] for _ in range(num_classes)] for _ in range(num_images)]
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
                    val_gt[i][int(c)].append(gt_box[j])
                for k, c in enumerate(all_pred_cls[b].cpu().numpy()):
                    pred_temp = {'image_id': i,
                                 'bbox': all_pred_boxes[b][k].cpu().numpy(),
                                 'scores': all_pred_scores[b][k].cpu().numpy()}
                    dete_boxes[int(c)].append(pred_temp)
                i = i + 1

    aps = []
    for i in range(1, num_classes):
        gt_roidb = [x[i] for x in val_gt]
        dete_roidb = dete_boxes[i]
        ap = evaluation(gt_roidb, dete_roidb, i, iou_thresh=0.5, use_07_metric=False)
        aps.append(ap)
    mAP = np.mean(aps)
    print('epoch: {:d}, mAP: {:4f}'.format(epoch, mAP))
    f = open(output_dir + '/vrd_object.txt', "a")
    f.write('vrd')
    f.write('    ')
    f.write('epoch: ')
    f.write(str(epoch).zfill(2))
    f.write('    ')
    f.write('lr: ')
    f.write('%.6f' % (lr))
    f.write('    ')
    f.write('mAP: ')
    f.write('%.4f' % (mAP))
    f.write('\n')
    f.close()


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.dataset == "VRD":
        args.imdb_name = "./process/vrd_object_roidb.npz"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '50']
        num_class = 101
        num_predicate = 70

    elif args.dataset == "VG":
        args.imdb_name = "./process/vg_object_roidb.npz"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '50']
        num_class = 151
        num_predicate = 50

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    load_roidb = np.load(args.imdb_name, allow_pickle=True)
    roidb_temp = load_roidb['roidb'][()]
    roidb = roidb_temp['train_roidb']
    data_size = np.int32(len(roidb))

    if data_size % args.batch_size:
        choose_size = np.array(np.arange(data_size))
        shuffle(choose_size)
        size = args.batch_size - data_size % args.batch_size
        for i in np.arange(size):
            roidb.append(roidb_temp['train_roidb'][choose_size[i]])

    val_roidb = roidb_temp['val_roidb']
    val_img_size = len(val_roidb)
    if val_img_size % args.batch_size:
        choose_size = np.array(np.arange(val_img_size))
        shuffle(choose_size)
        size = args.batch_size - val_img_size % args.batch_size
        for i in np.arange(size):
            val_roidb.append(roidb_temp['val_roidb'][choose_size[i]])

    train_size = len(roidb)
    val_size = len(val_roidb)
    print('{:d} roidb entries'.format(train_size))
    print('{:d} val_roidb entries'.format(val_size))
    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = roibatchLoader(roidb, args.batch_size, num_class, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers)

    val_dataset = roibatchLoader(val_roidb, args.batch_size, num_class, training=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=args.num_workers)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    im_scale = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

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
    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(num_class, pretrained=True, class_agnostic=args.class_agnostic )
    elif args.net == 'res50':
        fasterRCNN = resnet(num_class, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(num_class, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(num_class, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        lr = 0.00001
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.cuda:
        fasterRCNN.cuda()
    if args.optimizer == "adam":
        lr = args.lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        lr = args.lr
        optimizer = NewNoamOpt(
            torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM),
            max_lr=lr, warmup=1, batchsize=args.batch_size,
            decay_start=args.lr_decay_step,
            decay_gamma=args.lr_decay_gamma,
            datasize=train_size)
    # optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    if args.resume:
        load_name = os.path.join(output_dir,'faster_rcnn_{}_{}_{}.pth'.format(
            args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        #fasterRCNN.load_state_dict(checkpoint['model'])
        model_dict = fasterRCNN.state_dict()
        state_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict.keys()}
        # print(state_dict.keys())
        model_dict.update(state_dict)
        fasterRCNN.load_state_dict(model_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(train_size / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")

    for epoch in range(args.start_epoch, args.max_epochs):
        train_epoch(fasterRCNN, dataloader, optimizer, epoch, iters_per_epoch, output_dir)
        val_epoch(fasterRCNN, val_dataloader, epoch, val_img_size, num_class, output_dir)

    if args.use_tfboard:
        logger.close()