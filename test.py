# --------------------------------------------------------
# Test program satisfy for all the model
# 待验证
# modified by xmj
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
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
# from model.faster_rcnn.vgg16_bn import vgg16
from model.faster_rcnn.vgg16 import vgg16
from setproctitle import setproctitle

import pdb

xrange = range  # Python 3

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='dg_union', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='vgg16', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    # parser.add_argument('--load_dir', dest='load_dir',
    #                     help='directory to load models', default="models",
    #                     type=str)
    # parser.add_argument('--testpart', dest='testpart',
    #                   help='test_s or test_t or test_all', default="unioncityscape",
    #                   type=str)
    parser.add_argument('--model_dir', dest='model_dir',
                        help='directory to load models', default="models.pth",
                        type=str)
    parser.add_argument('--part', dest='part',
                        help='test_s or test_t or test_all', default="test_t",
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
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--model_name', dest='model_name',
                        help='model file name',
                        default='res101.bs1.pth', type=str)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    # start --------------- add by xmj ---------------------------
    parser.add_argument('--T_Set', dest='T_Set',
                        help='Target dataset',
                        default='cityscape', type=str)
    parser.add_argument('--T_Part', dest='T_Part',
                        help='test_s or test_t or test', default='test',
                        type=str)
    parser.add_argument('--T_Type', dest='T_Type',
                        help='TargetType', default="s1",
                        type=str)
    # Source_1
    parser.add_argument('--S1_Set', dest='S1_Set',
                        help='source dataset',
                        default='cityscape', type=str)
    parser.add_argument('--S1_Part', dest='S1_Part',
                        help='train_s or train_t or train', default='train',
                        type=str)
    parser.add_argument('--S1_Type', dest='S1_Type',
                        help='SourceType', default="s1",
                        type=str)
    # Source_2
    parser.add_argument('--S2_Set', dest='S2_Set',
                        help='source dataset',
                        default='cityscape', type=str)
    parser.add_argument('--S2_Part', dest='S2_Part',
                        help='train_s or train_t or train', default='train',
                        type=str)
    parser.add_argument('--S2_Type', dest='S2_Type',
                        help='SourceType', default="s1",
                        type=str)
    # Source_3
    parser.add_argument('--S3_Set', dest='S3_Set',
                        help='source dataset',
                        default='cityscape', type=str)
    parser.add_argument('--S3_Part', dest='S3_Part',
                        help='train_s or train_t or train', default='train',
                        type=str)
    parser.add_argument('--S3_Type', dest='S3_Type',
                        help='SourceType', default="s1",
                        type=str)
    
    parser.add_argument('--DataYear', dest='DataYear',
                        help='DataYear', default="2007",
                        type=str)
    parser.add_argument('--DaraType', dest='DataType',
                        help='DataType', default="s1",
                        type=str)
    # end ------------------ add by xmj --------------------------
    args = parser.parse_args()
    return args

def test_model(test_model_dir, args):

    print(">>test model start")
    # pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    # print(args.t_imdb_name)
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdb_name, False)

    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    load_name = test_model_dir

    # initialize the network here.
    if args.net == 'vgg16':
        fasterRCNN_t = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN_t = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN_t = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN_t = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN_t.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN_t.load_state_dict({k: v for k, v in checkpoint['model'].items() if k in fasterRCNN_t.state_dict()})
    # fasterRCNN_t.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']


    print('load model successfully!')
    # initialize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN_t.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = args.T_Part+'_faster_rcnn_10'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                            imdb.num_classes, training=False, normalize = False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0,
                                pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    fasterRCNN_t.eval()
    # fasterRCNN_t.train()
    empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

    for i in range(num_images):

        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN_t(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:,j]>thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                        for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
            .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

        if vis:
            cv2.imwrite('result.png', im2show)
            pdb.set_trace()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    map = imdb.evaluate_detections(all_boxes, output_dir)

    end = time.time()
    print("test time: %0.4fs" % (end - start))

    return map

if __name__ == '__main__':

    setproctitle("< xmj_test >")
    args = parse_args()

    # print('Called with args:')
    # print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[4,8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "dg_union":
        print('loading our dataset...........')
        # 数据集格式 -> 数据集合_标签参考集合_s?_2007_train
        # '{}_{}_{}_{}'.format(SourceSet, TargetSet, CoSet_1, type, year, split)
        args.t_imdb_name     = args.T_Set + '_' + args.S1_Set + '_' + args.S2_Set + '_' + args.T_Type + '_' + args.DataYear + '_' + args.T_Part
        args.s1_imdb_name    = args.S1_Set + '_' + args.T_Set + '_' + args.S2_Set + '_' + args.S1_Type + '_' + args.DataYear + '_' + args.S1_Part
        args.s2_imdb_name    = args.S2_Set + '_' + args.T_Set + '_' + args.S1_Set + '_' + args.S2_Type + '_' + args.DataYear + '_' + args.S2_Part
        args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "cityscape":
        print('loading our dataset...........')
        args.s_imdb_name = "cityscape_2007_train_s"
        args.t_imdb_name = "cityscape_2007_train_t"
        args.s_imdbtest_name="cityscape_2007_test_s"
        args.t_imdbtest_name="cityscape_2007_test_t"
        args.all_imdb_name="cityscape_2007_train_all"
        args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "dgunion":
        print('loading our dataset...........')
        args.t_imdb_name = args.testpart + "_s1_2007_test_2k" + args.part
        args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    lr = cfg.TRAIN.LEARNING_RATE
    momentum = cfg.TRAIN.MOMENTUM
    weight_decay = cfg.TRAIN.WEIGHT_DECAY

    test_model(args.model_dir, args)
