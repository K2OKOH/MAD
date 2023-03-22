# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
#
# region import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from traceback import print_tb

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
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader

from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
        adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn_MV_3.vgg16 import vgg16

from test import test_model

import setproctitle
# endregion

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='dg_union', type=str)
    parser.add_argument('--net', dest='net',
                      help='vgg16, res101',
                      default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=10, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)
    parser.add_argument('--mode', dest="mode", help='set program mode(train_model/test_model/train_patch/test_patch)', 
                        default='test_model', type=str)
    parser.add_argument('--model_dir', dest='model_dir',
                        help='directory to load models', default="models.pth",
                        type=str)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="./SaveFile/model",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of workers to load data',
                        default=0, type=int)
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
                        help='whether to perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.002, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=6, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--lamda', dest='lamda',
                      help='DA loss param',
                      default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--loadname', dest='loadname',
                        help='input loadname',
                        default='s_cityscape.pth')
    # log and display
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--log_flag', dest='log_flag', # add by xmj
                        help='1:batch_loss, 2:epoch_test',
                        default=0, type=int)
    
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
    # Target
    parser.add_argument('--T_Set', dest='T_Set',
                        help='Target dataset',
                        default='cityscape', type=str)
    parser.add_argument('--T_Part', dest='T_Part',
                        help='test_s or test_t or test', default='test',
                        type=str)
    parser.add_argument('--T_Type', dest='T_Type',
                        help='TargetType', default="s1",
                        type=str)
    # not change parameters
    parser.add_argument('--DataYear', dest='DataYear',
                        help='DataYear', default="2007",
                        type=str)
    parser.add_argument('--DaraType', dest='DataType',
                        help='DataType', default="s1",
                        type=str)
    # add by xmj
    parser.add_argument('--Mission', dest='Mission',
                        help='Mission name', default="unnamed",
                        type=str)


    args = parser.parse_args()
    return args

def draw_box(draw_box,image,color,width=4):
    image = image.data.cpu().numpy()

    if len(draw_box.size()) == 3:
        draw_box = draw_box[0]
    elif len(draw_box.size()) == 2:
        draw_box = draw_box

    if color == "RED":
        box_RGB = [1,0,0]
        print("\033[5;31mGT:\033[0m")
    elif color == "GREEN":
        box_RGB = [0,1,0]
        print("\033[5;32mATTACK:\033[0m")
    elif color == "BLUE":
        box_RGB = [0,0,1]
        print("\033[5;34mPATCH:\033[0m")
    elif color == "YELLOW":
        box_RGB = [1,1,0]
        print("\033[5;33mPRE:\033[0m")
        
    for i in range(draw_box.size(0)):
        box = draw_box[i]
        box = box.cpu().numpy()

        w_s = int(box[0])
        h_s = int(box[1])
        w_e = int(box[2])
        h_e = int(box[3])
        
        # R
        image[0, 0, h_s:h_e, w_s:w_s+width] = box_RGB[0]
        image[0, 0, h_s:h_e, w_e-width:w_e] = box_RGB[0]
        image[0, 0, h_s:h_s+width, w_s:w_e] = box_RGB[0]
        image[0, 0, h_e-width:h_e, w_s:w_e] = box_RGB[0]
        # G
        image[0, 1, h_s:h_e, w_s:w_s+width] = box_RGB[1]
        image[0, 1, h_s:h_e, w_e-width:w_e] = box_RGB[1]
        image[0, 1, h_s:h_s+width, w_s:w_e] = box_RGB[1]
        image[0, 1, h_e-width:h_e, w_s:w_e] = box_RGB[1]
        # B
        image[0, 2, h_s:h_e, w_s:w_s+width] = box_RGB[2]
        image[0, 2, h_s:h_e, w_e-width:w_e] = box_RGB[2]
        image[0, 2, h_s:h_s+width, w_s:w_e] = box_RGB[2]
        image[0, 2, h_e-width:h_e, w_s:w_e] = box_RGB[2]

        print("No.%d\ts: (%s,%s)\te: (%s,%s)" % (i,w_s,h_s,w_e,h_e))

    return torch.from_numpy(image)

class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

if __name__ == '__main__':

    # setup_seed(20)

    args = parse_args()
    setproctitle.setproctitle("%s" %(args.Mission))

    print('Called with args:')
    print(args)

    # region 读取参数
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_train"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "cityscape":
        print('loading our dataset...........')
        args.s_imdb_name = "cityscape_2007_train_s"
        args.t_imdb_name = "cityscape_2007_train_t"
        args.s_imdbtest_name="cityscape_2007_test_s"
        args.t_imdbtest_name="cityscape_2007_test_t"
        args.all_imdb_name="cityscape_2007_train_all"
        args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "dg_union":
        print('loading our dataset...........')
        # 数据集格式 -> 数据集合_标签参考集合_s?_2007_train
        # '{}_{}_{}_{}_{}'.format(SourceSet, TargetSet, CoSet_1, type, year, split)
        args.t_imdb_name = args.T_Set + '_' + args.S1_Set + '_' + args.S2_Set + '_' + args.T_Type  + '_' + args.DataYear + '_' + args.T_Part
        args.s1_imdb_name    = args.S1_Set + '_' + args.T_Set + '_' + args.S2_Set + '_' + args.S1_Type + '_' + args.DataYear + '_' + args.S1_Part
        args.s2_imdb_name    = args.S2_Set + '_' + args.T_Set + '_' + args.S1_Set + '_' + args.S2_Type + '_' + args.DataYear + '_' + args.S2_Part
        args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
    print(args.dataset)
    print(args.set_cfgs)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    #torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    # endregion

    # 加载当前日期
    M_D = time.strftime("(%b-%d[%H])", time.localtime())

    def train_model():
        print(">>train model start")
        # -- Note: Use validation set and disable the flipped to enable faster loading.
        cfg.TRAIN.USE_FLIPPED = True
        cfg.USE_GPU_NMS = args.cuda
        '''
        s_imdb        -> 实例化后的数据集       !! 例如 imdb = cityscape(train_s, 2007)
        s_roidb       -> 每张图片标注字典的列表  !! 例如 [{ 第一张图片的字典 },{ 第二张图片的字典 },{...}]
        s_ratio_list  -> 排列后的长宽比列表
        s_ratio_index -> 长宽比的次序
        '''
        # 读取域的数据
        imdb_s1, roidb_s1, ratio_list_s1, ratio_index_s1 = combined_roidb(args.s1_imdb_name)
        imdb_s2, roidb_s2, ratio_list_s2, ratio_index_s2 = combined_roidb(args.s2_imdb_name)
        
        train_size_s1 = len(roidb_s1)
        train_size_s2 = len(roidb_s2)

        print('s1: {:d} s2: {:d} roidb entries'.format(len(roidb_s1),len(roidb_s2)))

        # region 创建输出目录
        output_dir = args.save_dir + "/" + args.net + "-" + args.S1_Set + "-" + args.S2_Set + '/' + args.Mission
        if not os.path.exists(output_dir):
            print('No directory named: ' + output_dir)
            if (input('Create it? [y/n]') == 'y'):
                os.makedirs(output_dir)
            else:
                print('exit')
                sys.exit(0)
        # endregion

        # region 创建 log 目录
        if args.log_flag:
            if (not args.resume):
                log_dir = output_dir + '/log:' + M_D
                loss_log_dir = log_dir + '/loss_log.txt'
                epoch_test_log_dir = log_dir + '/epoch_test_log.txt'
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)
                with open(loss_log_dir,"w") as f: 
                    f.write("[Date: %s]\r" %M_D)   #这句话自带文件关闭功能，不需要再写f.close( )
                with open(epoch_test_log_dir,"w") as f: 
                    f.write("[Date: %s]\r" %M_D)   #这句话自带文件关闭功能，不需要再写f.close( )
            else:
                print('load log')
                log_dir = output_dir + '/log:' + M_D + '_r'
                loss_log_dir = log_dir + '/loss_log.txt'
                epoch_test_log_dir = log_dir + '/epoch_test_log.txt'
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)
        # endregion

        # region dataloader
        # S1
        sampler_batch_s1 = sampler(train_size_s1, args.batch_size)
        dataset_s1 = roibatchLoader(roidb_s1, ratio_list_s1, ratio_index_s1, args.batch_size, \
                                imdb_s1.num_classes, training=True)
        dataloader_s1 = torch.utils.data.DataLoader(dataset_s1, batch_size=args.batch_size,
                                    sampler=sampler_batch_s1, num_workers=args.num_workers)

        # S2
        sampler_batch_s2 = sampler(train_size_s2, args.batch_size)
        dataset_s2 = roibatchLoader(roidb_s2, ratio_list_s2, ratio_index_s2, args.batch_size, \
                                imdb_s2.num_classes, training=True)
        dataloader_s2 = torch.utils.data.DataLoader(dataset_s2, batch_size=args.batch_size,
                                    sampler=sampler_batch_s2, num_workers=args.num_workers)

        # initialize the tensor holder here.
        im_data_s1 = torch.FloatTensor(1)
        im_info_s1 = torch.FloatTensor(1)
        num_boxes_s1 = torch.LongTensor(1)
        gt_boxes_s1 = torch.FloatTensor(1)

        im_data_s2 = torch.FloatTensor(1)
        im_info_s2 = torch.FloatTensor(1)
        num_boxes_s2 = torch.LongTensor(1)
        gt_boxes_s2 = torch.FloatTensor(1)
        # ship to cuda
        if args.cuda:
            im_data_s1 = im_data_s1.cuda()
            im_info_s1 = im_info_s1.cuda()
            num_boxes_s1 = num_boxes_s1.cuda()
            gt_boxes_s1 = gt_boxes_s1.cuda()

            im_data_s2 = im_data_s2.cuda()
            im_info_s2 = im_info_s2.cuda()
            num_boxes_s2 = num_boxes_s2.cuda()
            gt_boxes_s2 = gt_boxes_s2.cuda()

        # make variable
        im_data_s1 = Variable(im_data_s1)
        im_info_s1 = Variable(im_info_s1)
        num_boxes_s1 = Variable(num_boxes_s1)
        gt_boxes_s1 = Variable(gt_boxes_s1)
        
        im_data_s2 = Variable(im_data_s2)
        im_info_s2 = Variable(im_info_s2)
        num_boxes_s2 = Variable(num_boxes_s2)
        gt_boxes_s2 = Variable(gt_boxes_s2)

        # endregion

        if args.cuda:
            cfg.CUDA = True

        # region 初始化网络
        # initialize the network here.
        if args.net == 'vgg16':
            fasterRCNN = vgg16(imdb_s1.classes, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            fasterRCNN = resnet(imdb_s1.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            fasterRCNN = resnet(imdb_s1.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res152':
            fasterRCNN = resnet(imdb_s1.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()

        fasterRCNN.create_architecture()
        # endregion

        lr = cfg.TRAIN.LEARNING_RATE
        lr = args.lr
        #tr_momentum = cfg.TRAIN.MOMENTUM
        #tr_momentum = args.momentum

        params = []
        for key, value in dict(fasterRCNN.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

        if args.optimizer == "adam":
            # lr = lr * 0.1
            optimizer = torch.optim.Adam(params)

        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

        if args.cuda:
            fasterRCNN.cuda()

        if args.resume:
            # load_name = os.path.join(output_dir,
            # 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
            load_name = os.path.join(output_dir, args.loadname)
            print("loading checkpoint %s" % (load_name))
            checkpoint = torch.load(load_name)
            args.session = checkpoint['session']
            args.start_epoch = checkpoint['epoch']
            fasterRCNN.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
            if 'pooling_mode' in checkpoint.keys():
                cfg.POOLING_MODE = checkpoint['pooling_mode']
                print("loaded checkpoint %s" % (load_name))

        if args.mGPUs:
            fasterRCNN = nn.DataParallel(fasterRCNN)

        # 计算epoch的迭代次数 (少等多)
        # train_size_s = max(train_size_s1, train_size_s2)
        # 计算epoch的迭代次数 (多等少)
        train_size = min(train_size_s1, train_size_s2)
        iters_per_epoch = int(train_size / args.batch_size)

        for epoch in range(args.start_epoch, args.max_epochs + 1):
            # setting to train mode

            if args.log_flag:
                with open(loss_log_dir,"a") as f: 
                    f.write("epoch: %d\r" % (epoch))   #这句话自带文件关闭功能，不需要再写f.close( )

            fasterRCNN.train()
            loss_temp = 0
            start = time.time()

            if epoch % (args.lr_decay_step + 1) == 0:
                adjust_learning_rate(optimizer, args.lr_decay_gamma)
                lr *= args.lr_decay_gamma

            # 准备域的迭代器
            data_iter_s1 = iter(dataloader_s1)
            data_iter_s2 = iter(dataloader_s2)

            # 进行batch迭代
            # for step in range(2):
            for step in range(iters_per_epoch):
                
                data_s1 = next(data_iter_s1)
                data_s2 = next(data_iter_s2)

                # 针对 D_1 的训练
                im_data_s1.data.resize_(data_s1[0].size()).copy_(data_s1[0])   #change holder size
                im_info_s1.data.resize_(data_s1[1].size()).copy_(data_s1[1])
                gt_boxes_s1.data.resize_(data_s1[2].size()).copy_(data_s1[2])
                num_boxes_s1.data.resize_(data_s1[3].size()).copy_(data_s1[3])
                
                # 针对 D_2 的训练
                im_data_s2.data.resize_(data_s2[0].size()).copy_(data_s2[0])   #change holder size
                im_info_s2.data.resize_(data_s2[1].size()).copy_(data_s2[1])
                gt_boxes_s2.data.resize_(data_s2[2].size()).copy_(data_s2[2])
                num_boxes_s2.data.resize_(data_s2[3].size()).copy_(data_s2[3])

                fasterRCNN.zero_grad()

                rois_s1, cls_prob_s1, bbox_pred_s1, rpn_loss_cls_s1, rpn_loss_bbox_s1, RCNN_loss_cls_s1, RCNN_loss_bbox_s1, rois_label_s1, \
                rois_s2, cls_prob_s2, bbox_pred_s2, rpn_loss_cls_s2, rpn_loss_bbox_s2, RCNN_loss_cls_s2, RCNN_loss_bbox_s2, rois_label_s2, \
                DA_img_loss_cls_s1, DA_img_loss_cls_s2, DA_ins_loss_cls_s1, DA_ins_loss_cls_s2, DA_cst_loss_s1, DA_cst_loss_s2, \
                img_mv_recon_loss, img_mv_cls_loss, img_mv_dis_loss, \
                ins_mv_recon_loss, ins_mv_cls_loss, ins_mv_dis_loss, \
                MV_cst_loss_s1, MV_cst_loss_s2 \
                        = fasterRCNN(   im_data_s1, im_info_s1, gt_boxes_s1, num_boxes_s1, \
                                        im_data_s2, im_info_s2, gt_boxes_s2, num_boxes_s2 \
                                    )

                loss_s1 = rpn_loss_cls_s1.mean() + rpn_loss_bbox_s1.mean() + RCNN_loss_cls_s1.mean() + RCNN_loss_bbox_s1.mean()
                loss_s2 = rpn_loss_cls_s2.mean() + rpn_loss_bbox_s2.mean() + RCNN_loss_cls_s2.mean() + RCNN_loss_bbox_s2.mean()
                # DA
                img_loss = DA_img_loss_cls_s1.mean() + DA_img_loss_cls_s2.mean()
                instance_loss = DA_ins_loss_cls_s1.mean() + DA_ins_loss_cls_s2.mean()
                cst_loss_loss = DA_cst_loss_s1.mean() + DA_cst_loss_s2.mean()
                # MV_1
                # loss_img_MV_cls = DA_img_loss_cls_s1_en.mean() + DA_img_loss_cls_s2_en.mean()
                # loss_img_MV_re = AC_s1.mean() + AC_s2.mean()
                # loss_ins_MV1_cls = DA_ins_loss_cls_ac1_s1.mean() + DA_ins_loss_cls_ac1_s2.mean()
                # loss_ins_MV1_re = AC_ins_s1.mean() + AC_ins_s2.mean()
                # loss_cst_MV1 = DA_cst_loss_ac1_s1.mean() + DA_cst_loss_ac1_s2.mean()
                # add
                loss_cr = loss_s1 + loss_s2

                da_loss = img_loss + instance_loss + cst_loss_loss
                
                img_MV_loss = 0.1 * img_mv_recon_loss + 0.1 * img_mv_cls_loss + 0.01 * img_mv_dis_loss
                ins_MV_loss = 0.1 * ins_mv_recon_loss + 0.1 * ins_mv_cls_loss + 0.01 * ins_mv_dis_loss
                # img_MV_loss = 0.1 * img_mv_recon_loss + 0.01 * img_mv_dis_loss
                # ins_MV_loss = 0.1 * ins_mv_recon_loss + 0.01 * ins_mv_dis_loss
                cst_MV_loss = 0.1 * (MV_cst_loss_s1 + MV_cst_loss_s2)

                da_MV_loss = img_MV_loss + ins_MV_loss + cst_MV_loss
                # da_MV_loss = img_MV_loss# + ins_MV_loss
                # da_MV_loss = ins_MV_loss

                # loss = loss_cr + args.lamda * da_loss + da_MV_loss
                # abl_bl
                loss = loss_cr + da_MV_loss * epoch * 0.12
                loss_temp += loss.item()

                # backward
                optimizer.zero_grad()
                loss.backward()
                if args.net == "vgg16":
                    clip_gradient(fasterRCNN, 10.)
                optimizer.step()

                # region show
                if step % args.disp_interval == 0:
                    end = time.time()
                    if step > 0:
                        loss_temp /= (args.disp_interval + 1)

                    if args.mGPUs:
                        loss_rpn_cls = rpn_loss_cls_s1.mean().item()
                        loss_rpn_box = rpn_loss_bbox_s1.mean().item()
                        loss_rcnn_cls = RCNN_loss_cls_s1.mean().item()
                        loss_rcnn_box = RCNN_loss_bbox_s1.mean().item()
                        fg_cnt = torch.sum(rois_label_s1.data.ne(0))
                        bg_cnt = rois_label_s1.data.numel() - fg_cnt
                    else:
                        loss_rpn_cls = rpn_loss_cls_s1.item()
                        loss_rpn_box = rpn_loss_bbox_s1.item()
                        loss_rcnn_cls = RCNN_loss_cls_s1.item()
                        loss_rcnn_box = RCNN_loss_bbox_s1.item()
                        fg_cnt = torch.sum(rois_label_s1.data.ne(0))
                        bg_cnt = rois_label_s1.data.numel() - fg_cnt

                    print('Mission:  \033[0;32m%s\033[0m \t[session %d][epoch %2d][iter %4d/%4d]'               % (args.Mission, args.session, epoch, step, iters_per_epoch))
                    print("\t\033[0;31mloss: %.4f\033[0m \t\tlr: %.2e \t\tfg/bg=(%d/%d) \t\ttime cost: %f"      % (loss_temp, lr, fg_cnt, bg_cnt, end-start))
                    print("\tloss_s1: %.4f \tloss_s2: %.4f \tda_loss %.4f"                                      % (loss_s1, loss_s2, da_loss))
                    print("\trpn_cls: %.4f \trpn_box: %.4f \trcnn_cls: %.4f \trcnn_box %.4f"                    % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                    print("\timg_loss: %.4f \tinstance_loss: %.4f \t\tcst_loss_loss %.4f"                       % (img_loss, instance_loss, cst_loss_loss))
                    print("\timg_recon: %.4f \timg_cls: %.4f, \timg_dis: %.4f"                                  % (img_mv_recon_loss, img_mv_cls_loss, img_mv_dis_loss))
                    print("\tins_recon: %.4f \tins_cls: %.4f, \tins_dis: %.4f"                                  % (ins_mv_recon_loss, ins_mv_cls_loss, ins_mv_dis_loss))

                    # loss write
                    if args.log_flag:
                        with open(loss_log_dir,"a") as f: 
                            f.write("  count: %s  \tloss: %f\r" % (step, loss_temp))   #这句话自带文件关闭功能，不需要再写f.close( )

                    loss_temp = 0
                    start = time.time()
                # endregion

            # region save
            if epoch <= args.max_epochs:
                save_name = os.path.join(output_dir, 'model_e{}.pth'.format(epoch))
                save_checkpoint({
                    'session': args.session,
                    'epoch': epoch + 1,
                    'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
                print('save model: {}'.format(save_name))

                # test
                map = test_model(save_name, args)
                # MAP write
                if args.log_flag:
                    with open(epoch_test_log_dir,"a") as f: 
                        f.write("  epoch: %s  \tmap: %f\r" % (epoch, map))   #这句话自带文件关闭功能，不需要再写f.close( )
            # endregion
    
    # train set
    if args.mode == "train_model":
        train_model()
    elif args.mode == "test_model":
        map = test_model(args.model_dir, args)
        print("\n\tMAP: %s" % map)