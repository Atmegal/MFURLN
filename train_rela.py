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
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from lib.roi_data_layer.roibatchLoader_rela import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list
from lib.model.utils.net_utils import  save_checkpoint, NewNoamOpt
from random import shuffle
from lib.model.faster_rcnn.vgg16_rela import vgg16
from lib.model.faster_rcnn.resnet import resnet
from lib.datasets.evaluator import evaluation
from lib.model.utils.funcs import all_recall
from tensorboardX import SummaryWriter

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
                        help='directory to save models', default="outputs",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=4, type=int)
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
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="adam", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.005, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=1, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.5, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--load', dest='load',
                        help='load checkpoint or not',
                        type=str)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=7, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=6999, type=int)
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
            sub_boxes.resize_(data[4].size()).copy_(data[4])
            obj_boxes.resize_(data[5].size()).copy_(data[5])
            rela_gt.resize_(data[6].size()).copy_(data[6])

        model.zero_grad()
        dete_loss, rela_loss = \
            model(im_data, im_info, gt_boxes, num_boxes, sub_boxes, obj_boxes, rela_gt)

        loss = dete_loss.mean() + rela_loss.mean()
        loss_temp += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(epoch)

        if step % args.disp_interval == 0:
            end = time.time()
            if step > 0:
                loss_temp /= (args.disp_interval + 1)

            dete_loss = dete_loss.item()
            rela_loss = rela_loss.item()
            print("[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e, dete_loss: %.4f, rela_loss: %.4f, time: %.2f" \
                  % (epoch, step, iters_per_epoch, loss_temp, optimizer.rate(epoch), dete_loss, rela_loss, end-start))
            if args.use_tfboard:
                info = {
                    'loss': loss_temp,
                    'dete_loss': dete_loss,
                    'rela_loss': rela_loss,
                }
                logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)
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
    lr = optimizer.param_groups[0]['lr']
    P_R50 = 0.0
    R_R50 = 0.0
    P_R100 = 0.0
    R_R100 = 0.0
    N_total = 0.0
    dete_boxes = [[] for _ in range(num_classes)]
    val_gt_boxes = [[[] for _ in range(num_classes)] for _ in range(num_images)]

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

            for j, c in enumerate(gt_class):
                val_gt_boxes[i][int(c)].append(gt_box[j])
            for k, c in enumerate(all_pred_cls.cpu().numpy()):
                pred_temp = {'image_id': i,
                             'bbox': all_pred_boxes[k].cpu().numpy(),
                             'scores': all_pred_scores[k].cpu().numpy()}
                dete_boxes[int(c)].append(pred_temp)

    aps = []
    for j in range(1, num_classes):
        gt_roidb = [x[j] for x in val_gt_boxes]
        dete_roidb = dete_boxes[j]
        ap = evaluation(gt_roidb, dete_roidb, j, iou_thresh=0.5, use_07_metric=False)
        aps.append(ap)

    mAP = np.mean(aps)
    P_R50 = P_R50 / N_total
    R_R50 = R_R50 / N_total
    P_R100 = P_R100 / N_total
    R_R100 = R_R100 / N_total
    print('epoch: {:d}, mAP: {:4f}, phrase_R50: {:.4f}, phrase_R100: {:.4f}, rela_R50: {:.4f}, rela_R100: {:.4f}'.
          format(epoch, mAP, P_R50, P_R100, R_R50, R_R100))
    f = open(output_dir+'/val.txt', "a")
    f.write('vrd')
    f.write('   ')
    f.write('epoch: ')
    f.write(str(epoch).zfill(2))
    f.write('   ')
    f.write('lr: ')
    f.write('%.6f' % (lr))
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


if __name__ == '__main__':

    args = parse_args()
    print('Called with args:')
    print(args)
    args.cuda = True
    if args.dataset == "VRD":
        args.imdb_name = "./process/vrd_rela_roidb.npz"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]',
                         'MAX_NUM_GT_BOXES', '50']
        num_class = 101
        num_predicate = 70
    elif args.dataset == "VG":
        args.imdb_name = "./process/vg_rela_roidb.npz"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
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

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    # imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    # load dataset
    load_roidb = np.load(args.imdb_name, allow_pickle=True)
    roidb_temp = load_roidb['roidb'][()]
    roidb = roidb_temp['train_roidb']
    test_roidb = roidb_temp['test_roidb']
    val_roidb = roidb_temp['val_roidb']
    train_size = len(roidb)
    val_size = len(val_roidb)
    print('{:d} train_roidb entries'.format(train_size))
    print('{:d} val_roidb entries'.format(val_size))

    output_dir =  "./outputs/relations/vrd/mfurln/"
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
    sub_boxes = torch.FloatTensor(1)
    obj_boxes = torch.FloatTensor(1)
    rela_gt = torch.FloatTensor(1)

    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        im_scale = im_scale.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        sub_boxes = sub_boxes.cuda()
        obj_boxes = obj_boxes.cuda()
        rela_gt = rela_gt.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    im_scale = Variable(im_scale)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    sub_boxes = Variable(sub_boxes)
    obj_boxes = Variable(obj_boxes)
    rela_gt = Variable(rela_gt)

    if args.cuda:
        cfg.CUDA = True
    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(num_class, num_predicate, pretrained=False,
                           class_agnostic=args.class_agnostic )
    elif args.net == 'res50':
        fasterRCNN = resnet(num_class, 50, num_predicate, pretrained=False,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(num_class, 101, num_predicate, pretrained=False,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(num_class, 152, num_predicate, pretrained=False,
                            class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()
    lr = 0.00001
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'rela' in key:
                print('train params:', key)
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
            else:
                value.requires_grad = False
                print('fixed params:', key)
    if args.cuda:
        fasterRCNN.cuda()

    if args.optimizer == "adam":
        lr = args.lr * 0.1
        optimizer = NewNoamOpt(
            torch.optim.Adam(params),
            max_lr=lr, warmup=1, batchsize=args.batch_size,
            decay_start=args.lr_decay_step,
            decay_gamma=args.lr_decay_gamma,
            datasize=train_size)
    elif args.optimizer == "sgd":
        lr = args.lr
        optimizer = NewNoamOpt(
            torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM),
            max_lr=lr, warmup=1, batchsize=args.batch_size,
            decay_start=args.lr_decay_step,
            decay_gamma=args.lr_decay_gamma,
            datasize=train_size)

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

    model_dict = fasterRCNN.state_dict()
    # args.load = './outputs/objects/vgg16/VRD/faster_rcnn_1_8_583.pth'
    load_model = torch.load(args.load)
    state_dict = {k: v for k, v in load_model['model'].items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    fasterRCNN.load_state_dict(model_dict)

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(train_size / args.batch_size)

    if args.use_tfboard:
        logger = SummaryWriter("logs")

    for epoch in range(args.start_epoch, args.max_epochs):
        train_epoch(fasterRCNN, dataloader, optimizer, epoch, iters_per_epoch, output_dir)
        val_epoch(fasterRCNN, val_dataloader, epoch, val_size, num_class, output_dir)

    if args.use_tfboard:
        logger.close()
