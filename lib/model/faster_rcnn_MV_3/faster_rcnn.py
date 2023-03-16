import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
# DA
from model.faster_rcnn_MV_3.DA import _ImageDA, _InstanceDA, _ImageDA_noGRL, _InstanceDA_En, _ImageDA_res
from model.faster_rcnn_MV_3.AC import AutoDecoder, AutoEncoder, InsEncoder, InsDecoder, ImgEncoder_1, ImgEncoder_2, ImgEncoder_3, ImgDecoder_1, ImgDecoder_2, ImgDecoder_3

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        # DA_s12
        self.RCNN_imageDA = _ImageDA(self.dout_base_model)
        self.RCNN_instanceDA = _InstanceDA()

        # DA_MV
        self.RCNN_imageDA_en1 = _ImageDA(128)    # 域判别头
        self.RCNN_imageDA_en2 = _ImageDA(128)    # 域判别头
        self.RCNN_imageDA_en3 = _ImageDA(128)    # 域判别头
        # self.RCNN_imageDA_en = _ImageDA_res(128)    # 域判别头
        # self.RCNN_imageDA_en1 = self.RCNN_imageDA_en    # 域判别头
        # self.RCNN_imageDA_en2 = self.RCNN_imageDA_en    # 域判别头
        # self.RCNN_imageDA_en3 = self.RCNN_imageDA_en    # 域判别头

        # self.RCNN_imageDA_en1 = _ImageDA_res(128)    # 域判别头
        # self.RCNN_imageDA_en2 = _ImageDA_res(128)    # 域判别头
        # self.RCNN_imageDA_en3 = _ImageDA_res(128)    # 域判别头
        
        # self.RCNN_insDA_or = _InstanceDA()
        self.RCNN_insDA_en1 = _InstanceDA_En()
        self.RCNN_insDA_en2 = _InstanceDA_En()
        self.RCNN_insDA_en3 = _InstanceDA_En()
        # self.RCNN_insDA_en = _InstanceDA_En()
        # self.RCNN_insDA_en1 = self.RCNN_insDA_en
        # self.RCNN_insDA_en2 = self.RCNN_insDA_en
        # self.RCNN_insDA_en3 = self.RCNN_insDA_en
        
        self.RCNN_insDA_loss = nn.BCELoss()

        self.consistency_loss = torch.nn.MSELoss(size_average=False)

        self.ln_img = nn.LayerNorm(normalized_shape = [10, 19])
        self.ln_ins = nn.LayerNorm(normalized_shape = [512])

        # AutoEncoder
        self.AutoEn_1 = ImgEncoder_1()
        self.AutoDe_1 = ImgDecoder_1()
        self.InsEn_1 = InsEncoder()
        self.InsDe_1 = InsDecoder()

        self.AutoEn_2 = ImgEncoder_2()
        self.AutoDe_2 = ImgDecoder_2()
        self.InsEn_2 = InsEncoder()
        self.InsDe_2 = InsDecoder()

        self.AutoEn_3 = ImgEncoder_3()
        self.AutoDe_3 = ImgDecoder_3()
        self.InsEn_3 = InsEncoder()
        self.InsDe_3 = InsDecoder()

        self.mse_loss = nn.MSELoss()
        self.upsample = torch.nn.Upsample(size=(40, 76), mode='bilinear')

    def forward(self,   im_data_s1, im_info_s1, gt_boxes_s1, num_boxes_s1,
                        im_data_s2, im_info_s2, gt_boxes_s2, num_boxes_s2 ):

        batch_size = im_data_s1.size(0)

        im_info_s1 = im_info_s1.data     #(size1,size2, image ratio(new image / source image) )
        gt_boxes_s1 = gt_boxes_s1.data
        num_boxes_s1 = num_boxes_s1.data
        domain_label_s1 = Variable(torch.FloatTensor([1.] * batch_size).cuda())

        im_info_s2 = im_info_s2.data     #(size1,size2, image ratio(new image / source image) )
        gt_boxes_s2 = gt_boxes_s2.data
        num_boxes_s2 = num_boxes_s2.data
        domain_label_s2 = Variable(torch.FloatTensor([0.] * batch_size).cuda())

        # feed image data to base model to obtain base feature map
        base_feat_s1 = self.RCNN_base(im_data_s1)
        base_feat_s2 = self.RCNN_base(im_data_s2)

        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.train()
        rois_s1, rpn_loss_cls_s1, rpn_loss_bbox_s1 = self.RCNN_rpn(base_feat_s1, im_info_s1, gt_boxes_s1, num_boxes_s1)
        rois_s2, rpn_loss_cls_s2, rpn_loss_bbox_s2 = self.RCNN_rpn(base_feat_s2, im_info_s2, gt_boxes_s2, num_boxes_s2)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            # s1
            roi_data_s1 = self.RCNN_proposal_target(rois_s1, gt_boxes_s1, num_boxes_s1)
            rois_s1, rois_label_s1, rois_target_s1, rois_inside_ws_s1, rois_outside_ws_s1 = roi_data_s1

            rois_label_s1 = Variable(rois_label_s1.view(-1).long())
            rois_target_s1 = Variable(rois_target_s1.view(-1, rois_target_s1.size(2)))
            rois_inside_ws_s1 = Variable(rois_inside_ws_s1.view(-1, rois_inside_ws_s1.size(2)))
            rois_outside_ws_s1 = Variable(rois_outside_ws_s1.view(-1, rois_outside_ws_s1.size(2)))

            # s2
            roi_data_s2 = self.RCNN_proposal_target(rois_s2, gt_boxes_s2, num_boxes_s2)
            rois_s2, rois_label_s2, rois_target_s2, rois_inside_ws_s2, rois_outside_ws_s2 = roi_data_s2

            rois_label_s2 = Variable(rois_label_s2.view(-1).long())
            rois_target_s2 = Variable(rois_target_s2.view(-1, rois_target_s2.size(2)))
            rois_inside_ws_s2 = Variable(rois_inside_ws_s2.view(-1, rois_inside_ws_s2.size(2)))
            rois_outside_ws_s2 = Variable(rois_outside_ws_s2.view(-1, rois_outside_ws_s2.size(2)))
            
        else:
            rois_label_s1 = None
            rois_target_s1 = None
            rois_inside_ws_s1 = None
            rois_outside_ws_s1 = None
            rpn_loss_cls_s1 = 0
            rpn_loss_bbox_s1 = 0

            rois_label_s2 = None
            rois_target_s2 = None
            rois_inside_ws_s2 = None
            rois_outside_ws_s2 = None
            rpn_loss_cls_s2 = 0
            rpn_loss_bbox_s2 = 0

        rois_s1 = Variable(rois_s1)
        rois_s2 = Variable(rois_s2)
        # do roi pooling based on predicted rois

        # roi pooling
        if cfg.POOLING_MODE == 'align':
            pooled_feat_s1 = self.RCNN_roi_align(base_feat_s1, rois_s1.view(-1, 5))
            pooled_feat_s2 = self.RCNN_roi_align(base_feat_s2, rois_s2.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat_s1 = self.RCNN_roi_pool(base_feat_s1, rois_s1.view(-1,5))
            pooled_feat_s2 = self.RCNN_roi_pool(base_feat_s2, rois_s2.view(-1,5))

        # feed pooled features to top model
        pooled_feat_s1 = self._head_to_tail(pooled_feat_s1)
        pooled_feat_s2 = self._head_to_tail(pooled_feat_s2)

        # compute bbox offset
        bbox_pred_s1 = self.RCNN_bbox_pred(pooled_feat_s1)
        bbox_pred_s2 = self.RCNN_bbox_pred(pooled_feat_s2)

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            # s1
            bbox_pred_view_s1 = bbox_pred_s1.view(bbox_pred_s1.size(0), int(bbox_pred_s1.size(1) / 4), 4)
            bbox_pred_select_s1 = torch.gather(bbox_pred_view_s1, 1, rois_label_s1.view(rois_label_s1.size(0), 1, 1).expand(rois_label_s1.size(0), 1, 4))
            bbox_pred_s1 = bbox_pred_select_s1.squeeze(1)
            # s2
            bbox_pred_view_s2 = bbox_pred_s2.view(bbox_pred_s2.size(0), int(bbox_pred_s2.size(1) / 4), 4)
            bbox_pred_select_s2 = torch.gather(bbox_pred_view_s2, 1, rois_label_s2.view(rois_label_s2.size(0), 1, 1).expand(rois_label_s2.size(0), 1, 4))
            bbox_pred_s2 = bbox_pred_select_s2.squeeze(1)

        # compute object classification probability
        cls_score_s1 = self.RCNN_cls_score(pooled_feat_s1)
        cls_prob_s1 = F.softmax(cls_score_s1, 1)
        cls_score_s2 = self.RCNN_cls_score(pooled_feat_s2)
        cls_prob_s2 = F.softmax(cls_score_s2, 1)

        RCNN_loss_cls_s1 = 0
        RCNN_loss_bbox_s1 = 0
        RCNN_loss_cls_s2 = 0
        RCNN_loss_bbox_s2 = 0

        if self.training:
            # classification loss
            RCNN_loss_cls_s1 = F.cross_entropy(cls_score_s1, rois_label_s1)
            RCNN_loss_cls_s2 = F.cross_entropy(cls_score_s2, rois_label_s2)

            # bounding box regression L1 loss
            RCNN_loss_bbox_s1 = _smooth_l1_loss(bbox_pred_s1, rois_target_s1, rois_inside_ws_s1, rois_outside_ws_s1)
            RCNN_loss_bbox_s2 = _smooth_l1_loss(bbox_pred_s2, rois_target_s2, rois_inside_ws_s2, rois_outside_ws_s2)

        cls_prob_s1 = cls_prob_s1.view(batch_size, rois_s1.size(1), -1)
        bbox_pred_s1 = bbox_pred_s1.view(batch_size, rois_s1.size(1), -1)
        cls_prob_s2 = cls_prob_s2.view(batch_size, rois_s2.size(1), -1)
        bbox_pred_s2 = bbox_pred_s2.view(batch_size, rois_s2.size(1), -1)

        # region [DA loss]
        """ start -------------------------- DA loss ---------------------------- """
        """ start ----------- image level ------------- """
        # region [image level]
        DA_img_loss_cls_s1 = 0
        DA_img_loss_cls_s2 = 0

        # region [img 单视角]
        base_score_s1, base_label_s1 = self.RCNN_imageDA(base_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        base_score_s2, base_label_s2 = self.RCNN_imageDA(base_feat_s2, Variable(torch.FloatTensor([1.] * batch_size).cuda()))

        base_prob_s1 = F.log_softmax(base_score_s1, dim=1)
        DA_img_loss_cls_s1 = F.nll_loss(base_prob_s1, base_label_s1)

        base_prob_s2 = F.log_softmax(base_score_s2, dim=1)
        DA_img_loss_cls_s2 = F.nll_loss(base_prob_s2, base_label_s2)
        # endregion

        # region [img MV]

        # 上采样
        base_feat_s1_re = self.upsample(base_feat_s1)
        base_feat_s2_re = self.upsample(base_feat_s2)

        # region [视角1]
        MV1_feat_s1 = self.AutoEn_1(base_feat_s1_re)
        MV1_feat_s2 = self.AutoEn_1(base_feat_s2_re)
        re1_feat_s1 = self.AutoDe_1(MV1_feat_s1)
        re1_feat_s2 = self.AutoDe_1(MV1_feat_s2)
        # 归一化
        MV1_feat_s1 = self.ln_img(MV1_feat_s1)
        MV1_feat_s2 = self.ln_img(MV1_feat_s2)
        # 判别器损失
        MV1_score_s1, MV1_label_s1 = self.RCNN_imageDA_en1(MV1_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        MV1_score_s2, MV1_label_s2 = self.RCNN_imageDA_en1(MV1_feat_s2, Variable(torch.FloatTensor([1.] * batch_size).cuda()))
        
        MV1_prob_s1_en = F.log_softmax(MV1_score_s1, dim=1)
        DA_img_MV1_s1 = F.nll_loss(MV1_prob_s1_en, MV1_label_s1)

        MV1_prob_s2_en = F.log_softmax(MV1_score_s2, dim=1)
        DA_img_MV1_s2 = F.nll_loss(MV1_prob_s2_en, MV1_label_s2)

        # recon_loss
        MV1_s1 = self.mse_loss(re1_feat_s1, base_feat_s1_re.detach())
        MV1_s2 = self.mse_loss(re1_feat_s2, base_feat_s2_re.detach())

        # 重构损失 + 域分类损失
        # img_MV1_loss = MV1_s1 + MV1_s2 + DA_img_MV1_s1 + DA_img_MV1_s2
        img_MV1_recon_loss = MV1_s1 + MV1_s2
        img_MV1_cls_loss = DA_img_MV1_s1 + DA_img_MV1_s2

        # endregion

        # region [视角2]
        MV2_feat_s1 = self.AutoEn_2(base_feat_s1_re)
        MV2_feat_s2 = self.AutoEn_2(base_feat_s2_re)
        re2_feat_s1 = self.AutoDe_2(MV2_feat_s1)
        re2_feat_s2 = self.AutoDe_2(MV2_feat_s2)
        # 归一化
        MV2_feat_s1 = self.ln_img(MV2_feat_s1)
        MV2_feat_s2 = self.ln_img(MV2_feat_s2)
        # 判别器损失
        MV2_score_s1, MV2_label_s1_en = self.RCNN_imageDA_en2(MV2_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        MV2_score_s2, MV2_label_s2_en = self.RCNN_imageDA_en2(MV2_feat_s2, Variable(torch.FloatTensor([1.] * batch_size).cuda()))
        
        MV2_prob_s1_en = F.log_softmax(MV2_score_s1, dim=1)
        DA_img_MV2_s1 = F.nll_loss(MV2_prob_s1_en, MV2_label_s1_en)

        MV2_prob_s2_en = F.log_softmax(MV2_score_s2, dim=1)
        DA_img_MV2_s2 = F.nll_loss(MV2_prob_s2_en, MV2_label_s2_en)

        # recon_loss
        MV2_s1 = self.mse_loss(re2_feat_s1, base_feat_s1_re.detach())
        MV2_s2 = self.mse_loss(re2_feat_s2, base_feat_s2_re.detach())

        # 重构损失 + 域分类损失
        # img_MV2_loss = MV2_s1 + MV2_s2 + DA_img_MV2_s1 + DA_img_MV2_s2
        img_MV2_recon_loss = MV2_s1 + MV2_s2
        img_MV2_cls_loss = DA_img_MV2_s1 + DA_img_MV2_s2
        
        # endregion

        # region [视角3]
        MV3_feat_s1 = self.AutoEn_3(base_feat_s1_re)
        MV3_feat_s2 = self.AutoEn_3(base_feat_s2_re)
        re3_feat_s1 = self.AutoDe_3(MV3_feat_s1)
        re3_feat_s2 = self.AutoDe_3(MV3_feat_s2)
        # 归一化
        MV3_feat_s1 = self.ln_img(MV3_feat_s1)
        MV3_feat_s2 = self.ln_img(MV3_feat_s2)
        # 判别器损失
        MV3_score_s1, MV3_label_s1_en = self.RCNN_imageDA_en3(MV3_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        MV3_score_s2, MV3_label_s2_en = self.RCNN_imageDA_en3(MV3_feat_s2, Variable(torch.FloatTensor([1.] * batch_size).cuda()))
        
        MV3_prob_s1_en = F.log_softmax(MV3_score_s1, dim=1)
        DA_img_MV3_s1 = F.nll_loss(MV3_prob_s1_en, MV3_label_s1_en)

        MV3_prob_s2_en = F.log_softmax(MV3_score_s2, dim=1)
        DA_img_MV3_s2 = F.nll_loss(MV3_prob_s2_en, MV3_label_s2_en)

        # recon_loss
        MV3_s1 = self.mse_loss(re3_feat_s1, base_feat_s1_re.detach())
        MV3_s2 = self.mse_loss(re3_feat_s2, base_feat_s2_re.detach())

        # 重构损失 + 域分类损失
        img_MV3_recon_loss = MV3_s1 + MV3_s2
        img_MV3_cls_loss = DA_img_MV3_s1 + DA_img_MV3_s2
        
        # endregion

        dif12_s1 = (self.mse_loss(MV1_feat_s1, MV2_feat_s1.detach()) + self.mse_loss(MV2_feat_s1, MV1_feat_s1.detach()))/2
        dif12_s2 = (self.mse_loss(MV1_feat_s2, MV2_feat_s2.detach()) + self.mse_loss(MV2_feat_s2, MV1_feat_s2.detach()))/2

        dif13_s1 = (self.mse_loss(MV1_feat_s1, MV3_feat_s1.detach()) + self.mse_loss(MV3_feat_s1, MV1_feat_s1.detach()))/2
        dif13_s2 = (self.mse_loss(MV1_feat_s2, MV3_feat_s2.detach()) + self.mse_loss(MV3_feat_s2, MV1_feat_s2.detach()))/2

        dif23_s1 = (self.mse_loss(MV3_feat_s1, MV2_feat_s1.detach()) + self.mse_loss(MV2_feat_s1, MV3_feat_s1.detach()))/2
        dif23_s2 = (self.mse_loss(MV3_feat_s2, MV2_feat_s2.detach()) + self.mse_loss(MV2_feat_s2, MV3_feat_s2.detach()))/2

        img_mv_dis_loss = torch.exp(-(dif12_s1 + dif12_s2 + dif13_s1 + dif13_s2 + dif23_s1 + dif23_s2))
        # img_MV_loss = img_MV1_loss + img_MV2_loss + img_MV3_loss + un_dis # - (0.01) * (dif_s1 + dif_s2)
        img_mv_recon_loss = img_MV3_recon_loss + img_MV2_recon_loss + img_MV1_recon_loss
        img_mv_cls_loss = img_MV3_cls_loss + img_MV2_cls_loss + img_MV1_cls_loss

        # endregion

        # endregion

        """ start ----------- instance level ------------- """
        # region [instance level]
        
        # region [ins 单视角]
        DA_ins_loss_cls_s1 = 0
        DA_ins_loss_cls_s2 = 0
        # s1
        instance_sigmoid_s1, same_size_label_s1 = self.RCNN_instanceDA(pooled_feat_s1, domain_label_s1)
        instance_loss_s1 = nn.BCELoss()
        DA_ins_loss_cls_s1 = instance_loss_s1(instance_sigmoid_s1, same_size_label_s1)
        # s2
        instance_sigmoid_s2, same_size_label_s2 = self.RCNN_instanceDA(pooled_feat_s2, domain_label_s2)
        instance_loss_s2 = nn.BCELoss()
        DA_ins_loss_cls_s2 = instance_loss_s2(instance_sigmoid_s2, same_size_label_s2)
        # endregion 

        # region [ins MV]

        # region [视角1]
        MV1_ins_feat_s1 = self.InsEn_1(pooled_feat_s1)
        MV1_ins_feat_s2 = self.InsEn_1(pooled_feat_s2)
        re1_ins_feat_s1 = self.InsDe_1(MV1_ins_feat_s1)
        re1_ins_feat_s2 = self.InsDe_1(MV1_ins_feat_s2)
        # 归一化
        MV1_ins_feat_s1 = self.ln_ins(MV1_ins_feat_s1)
        MV1_ins_feat_s2 = self.ln_ins(MV1_ins_feat_s2)

        # 1. recon_loss(重构损失)
        MV1_ins_s1 = self.mse_loss(re1_ins_feat_s1, pooled_feat_s1.detach())
        MV1_ins_s2 = self.mse_loss(re1_ins_feat_s2, pooled_feat_s2.detach())

        # 2. 领域分类损失
        # s1
        instance_sigmoid_s1_MV1, same_size_label_s1_MV1 = self.RCNN_insDA_en1(MV1_ins_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        DA_ins_MV1_s1 = self.RCNN_insDA_loss(instance_sigmoid_s1_MV1, same_size_label_s1_MV1)
        # s2
        instance_sigmoid_s2_MV1, same_size_label_s2_MV1 = self.RCNN_insDA_en1(MV1_ins_feat_s2, Variable(torch.FloatTensor([1.] * batch_size).cuda()))
        DA_ins_MV1_s2 = self.RCNN_insDA_loss(instance_sigmoid_s2_MV1, same_size_label_s2_MV1)

        # ins_MV1_loss = DA_ins_MV1_s1 + DA_ins_MV1_s2 + MV1_ins_s1 + MV1_ins_s2
        ins_MV1_recon_loss = MV1_ins_s1 + MV1_ins_s2
        ins_MV1_cls_loss = DA_ins_MV1_s1 + DA_ins_MV1_s2

        # endregion

        # region [视角2]
        MV2_ins_feat_s1 = self.InsEn_2(pooled_feat_s1)
        MV2_ins_feat_s2 = self.InsEn_2(pooled_feat_s2)
        re2_ins_feat_s1 = self.InsDe_2(MV2_ins_feat_s1)
        re2_ins_feat_s2 = self.InsDe_2(MV2_ins_feat_s2)
        # 归一化
        MV2_ins_feat_s1 = self.ln_ins(MV2_ins_feat_s1)
        MV2_ins_feat_s2 = self.ln_ins(MV2_ins_feat_s2)

        # 1. recon_loss(重构损失)
        MV2_ins_s1 = self.mse_loss(re2_ins_feat_s1, pooled_feat_s1.detach())
        MV2_ins_s2 = self.mse_loss(re2_ins_feat_s2, pooled_feat_s2.detach())

        # 2. 领域分类损失
        # s1
        instance_sigmoid_s1_MV2, same_size_label_s1_MV2 = self.RCNN_insDA_en2(MV2_ins_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        DA_ins_MV2_s1 = self.RCNN_insDA_loss(instance_sigmoid_s1_MV2, same_size_label_s1_MV2)
        # s2
        instance_sigmoid_s2_MV2, same_size_label_s2_MV2 = self.RCNN_insDA_en2(MV2_ins_feat_s2, Variable(torch.FloatTensor([1.] * batch_size).cuda()))
        DA_ins_MV2_s2 = self.RCNN_insDA_loss(instance_sigmoid_s2_MV2, same_size_label_s2_MV2)

        # ins_MV2_loss = DA_ins_MV2_s1 + DA_ins_MV2_s2 + MV2_ins_s1 + MV2_ins_s2
        ins_MV2_recon_loss = MV2_ins_s1 + MV2_ins_s2
        ins_MV2_cls_loss = DA_ins_MV2_s1 + DA_ins_MV2_s2

        # endregion

        # region [视角3]
        MV3_ins_feat_s1 = self.InsEn_3(pooled_feat_s1)
        MV3_ins_feat_s2 = self.InsEn_3(pooled_feat_s2)
        re3_ins_feat_s1 = self.InsDe_3(MV3_ins_feat_s1)
        re3_ins_feat_s2 = self.InsDe_3(MV3_ins_feat_s2)
        # 归一化
        MV3_ins_feat_s1 = self.ln_ins(MV3_ins_feat_s1)
        MV3_ins_feat_s2 = self.ln_ins(MV3_ins_feat_s2)

        # 1. recon_loss(重构损失)
        MV3_ins_s1 = self.mse_loss(re3_ins_feat_s1, pooled_feat_s1.detach())
        MV3_ins_s2 = self.mse_loss(re3_ins_feat_s2, pooled_feat_s2.detach())

        # 2. 领域分类损失
        # s1
        instance_sigmoid_s1_MV3, same_size_label_s1_MV3 = self.RCNN_insDA_en3(MV3_ins_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        DA_ins_MV3_s1 = self.RCNN_insDA_loss(instance_sigmoid_s1_MV3, same_size_label_s1_MV3)
        # s2
        instance_sigmoid_s2_MV3, same_size_label_s2_MV3 = self.RCNN_insDA_en3(MV3_ins_feat_s2, Variable(torch.FloatTensor([1.] * batch_size).cuda()))
        DA_ins_MV3_s2 = self.RCNN_insDA_loss(instance_sigmoid_s2_MV3, same_size_label_s2_MV3)

        # ins_MV3_loss = DA_ins_MV3_s1 + DA_ins_MV3_s2 + MV3_ins_s1 + MV3_ins_s2
        ins_MV3_recon_loss = MV3_ins_s1 + MV3_ins_s2
        ins_MV3_cls_loss = DA_ins_MV3_s1 + DA_ins_MV3_s2

        # endregion

        # 3. 多视角损失
        dif12_ins_s1 = (self.mse_loss(MV1_ins_feat_s1, MV2_ins_feat_s1.detach()) + self.mse_loss(MV2_ins_feat_s1, MV1_ins_feat_s1.detach()))/2
        dif12_ins_s2 = (self.mse_loss(MV1_ins_feat_s2, MV2_ins_feat_s2.detach()) + self.mse_loss(MV2_ins_feat_s2, MV1_ins_feat_s2.detach()))/2

        dif13_ins_s1 = (self.mse_loss(MV1_ins_feat_s1, MV3_ins_feat_s1.detach()) + self.mse_loss(MV3_ins_feat_s1, MV1_ins_feat_s1.detach()))/2
        dif13_ins_s2 = (self.mse_loss(MV1_ins_feat_s2, MV3_ins_feat_s2.detach()) + self.mse_loss(MV3_ins_feat_s2, MV1_ins_feat_s2.detach()))/2
        
        dif23_ins_s1 = (self.mse_loss(MV3_ins_feat_s1, MV2_ins_feat_s1.detach()) + self.mse_loss(MV2_ins_feat_s1, MV3_ins_feat_s1.detach()))/2
        dif23_ins_s2 = (self.mse_loss(MV3_ins_feat_s2, MV2_ins_feat_s2.detach()) + self.mse_loss(MV2_ins_feat_s2, MV3_ins_feat_s2.detach()))/2
        
        ins_mv_dis_loss = torch.exp(-(dif12_ins_s1 + dif12_ins_s2 + dif13_ins_s1 + dif13_ins_s2 + dif23_ins_s1 + dif23_ins_s2))
        # ins_mv_dis_loss = 1/(dif12_ins_s1 + dif12_ins_s2 + dif13_ins_s1 + dif13_ins_s2 + dif23_ins_s1 + dif23_ins_s2)

        # ins_MV_loss = ins_MV1_loss + ins_MV2_loss + ins_MV3_loss + un_ins_dis # - (0.01) * (dif_ins_s1 + dif_ins_s2)
        ins_mv_recon_loss = ins_MV3_recon_loss + ins_MV2_recon_loss + ins_MV1_recon_loss
        ins_mv_cls_loss = ins_MV3_cls_loss + ins_MV2_cls_loss + ins_MV1_cls_loss

        # endregion

        # endregion

        """ start ----------- consistency ------------- """
        # region [consistency level]
        consistency_prob_s1 = F.softmax(base_score_s1, dim=1)[:,1,:,:]
        consistency_prob_s1 = torch.mean(consistency_prob_s1)
        consistency_prob_s1 = consistency_prob_s1.repeat(instance_sigmoid_s1.size())
        DA_cst_loss_s1 = self.consistency_loss(instance_sigmoid_s1, consistency_prob_s1.detach())
        
        consistency_prob_s2 = F.softmax(base_score_s2, dim=1)[:,0,:,:]
        consistency_prob_s2 = torch.mean(consistency_prob_s2)
        consistency_prob_s2 = consistency_prob_s2.repeat(instance_sigmoid_s2.size())
        DA_cst_loss_s2 = self.consistency_loss(instance_sigmoid_s2, consistency_prob_s2.detach())

        # soft cst
        MV_score_s1 = (MV1_score_s1 + MV2_score_s1 + MV3_score_s1)/3
        MV_score_s2 = (MV1_score_s2 + MV2_score_s2 + MV3_score_s2)/3

        instance_sigmoid_s1_MV = (instance_sigmoid_s1_MV1 + instance_sigmoid_s1_MV2 + instance_sigmoid_s1_MV3)/3
        instance_sigmoid_s2_MV = (instance_sigmoid_s2_MV1 + instance_sigmoid_s2_MV2 + instance_sigmoid_s2_MV3)/3

        soft_cst_prob_s1 = F.softmax(MV_score_s1, dim=1)[:,1,:,:]
        soft_cst_prob_s1 = torch.mean(soft_cst_prob_s1)
        soft_cst_prob_s1 = soft_cst_prob_s1.repeat(instance_sigmoid_s1_MV.size())
        MV_cst_loss_s1 = self.consistency_loss(instance_sigmoid_s1_MV, soft_cst_prob_s1.detach())

        soft_cst_s2 = F.softmax(MV_score_s2, dim=1)[:,0,:,:]
        soft_cst_s2 = torch.mean(soft_cst_s2)
        soft_cst_s2 = soft_cst_s2.repeat(instance_sigmoid_s2_MV.size())
        MV_cst_loss_s2 = self.consistency_loss(instance_sigmoid_s2_MV, soft_cst_s2.detach())
        
        # endregion

        # endregion
        return  rois_s1, cls_prob_s1, bbox_pred_s1, rpn_loss_cls_s1, rpn_loss_bbox_s1, RCNN_loss_cls_s1, RCNN_loss_bbox_s1, rois_label_s1, \
                rois_s2, cls_prob_s2, bbox_pred_s2, rpn_loss_cls_s2, rpn_loss_bbox_s2, RCNN_loss_cls_s2, RCNN_loss_bbox_s2, rois_label_s2, \
                DA_img_loss_cls_s1, DA_img_loss_cls_s2, DA_ins_loss_cls_s1, DA_ins_loss_cls_s2, DA_cst_loss_s1, DA_cst_loss_s2, \
                img_mv_recon_loss, img_mv_cls_loss, img_mv_dis_loss, \
                ins_mv_recon_loss, ins_mv_cls_loss, ins_mv_dis_loss, \
                MV_cst_loss_s1, MV_cst_loss_s2
                
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
