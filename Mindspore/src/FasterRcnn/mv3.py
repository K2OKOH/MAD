# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""FasterRcnn Rcnn network."""

import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter

class AutoEncoder(nn.Cell):
    def __init__(self, channel):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.SequentialCell(
            # Encoder
            # input (b, 512, 40, 76)
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, pad_mode="pad"),
            # nn.BatchNorm2d(128),
            # nn.ReLU()
            # output (b, 128, 10, 19)
        )

    def construct(self, *input):
        out = self.encoder(*input)
        return out

class AutoDecoder(nn.Cell):
    def __init__(self, channel):
        super(AutoDecoder, self).__init__()

        self.decoder = nn.SequentialCell(
            # DEcoder
            # nn.Conv2dTranspose(128, 256, kernel_size=3, stride=2, padding=1, pad_mode="pad"),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.Conv2dTranspose(128, channel, kernel_size=3, stride=1, padding=1, pad_mode="pad"),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )

    def construct(self, *input):
        out = self.decoder(*input)
        return out

class DenseNoTranpose(nn.Cell):
    """Dense method"""

    def __init__(self, input_channels, output_channels, weight_init):
        super(DenseNoTranpose, self).__init__()
        self.weight = Parameter(ms.common.initializer.initializer(weight_init, [input_channels, output_channels], ms.float32))
        self.bias = Parameter(ms.common.initializer.initializer("zeros", [output_channels], ms.float32))

        self.matmul = ops.MatMul(transpose_b=False)
        self.bias_add = ops.BiasAdd()
        self.cast = ops.Cast()
        self.device_type = "Ascend" if ms.get_context("device_target") == "Ascend" else "Others"

    def construct(self, x):
        if self.device_type == "Ascend":
            x = self.cast(x, ms.float16)
            weight = self.cast(self.weight, ms.float16)
            output = self.bias_add(self.matmul(x, weight), self.bias)
        else:
            output = self.bias_add(self.matmul(x, self.weight), self.bias)
        return output

class GradReverse(nn.Cell):
    def __init__(self, lambd):
        super(GradReverse, self).__init__()
        self.lambd = lambd

    def construct(self, x):
        return x

    def bprop(self, x, out, grad_output):
        return (grad_output * -self.lambd)
    
class _ImageDA(nn.Cell):
    def __init__(self,dim):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 64, kernel_size=1, stride=1)
        # self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.Conv2=nn.Conv2d(64,2,kernel_size=1,stride=1)
        self.reLu=nn.ReLU()
        # self.LabelResizeLayer=ImageLabelResizeLayer()
        self.grl = GradReverse(0.1)

    def construct(self, x):
        x=self.grl(x)
        x=self.reLu(self.Conv1(x))
        x=self.Conv2(x)

        return x

class Ins_D(nn.Cell):
    def __init__(self,
                config,
                representation_size,
                batch_size,
                num_classes,
                ):
        super(Ins_D, self).__init__()
        cfg = config
        self.dtype = np.float32
        self.ms_type = ms.float32
        self.rcnn_fc_out_channels = cfg.rcnn_fc_out_channels
        self.num_classes = num_classes
        self.num_classes_fronted = num_classes
        self.in_channels = cfg.rcnn_in_channels
        self.train_batch_size = batch_size
        self.test_batch_size = cfg.test_batch_size

        shape_0 = (self.rcnn_fc_out_channels, representation_size)
        weights_0 = ms.common.initializer.initializer("XavierUniform", shape=shape_0[::-1], dtype=self.ms_type).init_data()
        shape_1 = (self.rcnn_fc_out_channels, self.rcnn_fc_out_channels)
        weights_1 = ms.common.initializer.initializer("XavierUniform", shape=shape_1[::-1], dtype=self.ms_type).init_data()
        self.shared_fc_0 = DenseNoTranpose(representation_size, self.rcnn_fc_out_channels, weights_0)
        self.shared_fc_1 = DenseNoTranpose(self.rcnn_fc_out_channels, self.rcnn_fc_out_channels, weights_1)

        self.En1 = nn.Dense(self.rcnn_fc_out_channels, self.rcnn_fc_out_channels)
        self.En2 = nn.Dense(self.rcnn_fc_out_channels, self.rcnn_fc_out_channels)
        self.En3 = nn.Dense(self.rcnn_fc_out_channels, self.rcnn_fc_out_channels)

        self.De1 = nn.Dense(self.rcnn_fc_out_channels, self.rcnn_fc_out_channels)
        self.De2 = nn.Dense(self.rcnn_fc_out_channels, self.rcnn_fc_out_channels)
        self.De3 = nn.Dense(self.rcnn_fc_out_channels, self.rcnn_fc_out_channels)

        cls_weight = ms.common.initializer.initializer('Normal', shape=[num_classes, self.rcnn_fc_out_channels][::-1], dtype=self.ms_type).init_data()
        self.dc1_scores = DenseNoTranpose(self.rcnn_fc_out_channels, num_classes, cls_weight)
        self.dc2_scores = DenseNoTranpose(self.rcnn_fc_out_channels, num_classes, cls_weight)
        self.dc3_scores = DenseNoTranpose(self.rcnn_fc_out_channels, num_classes, cls_weight)

        self.flatten = ops.Flatten()
        self.relu = ops.ReLU()
        self.loss_dom = ops.SoftmaxCrossEntropyWithLogits()
        self.ce_loss  =  nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.onehot = ops.OneHot()
        self.cast = ops.Cast()

        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.zeros = ops.Zeros()
        self.ones = ops.Ones()
        
        self.grl = GradReverse(0.2)
        self.mse = nn.MSELoss(reduction='mean')
        
    def construct(self, feat_s1, feat_s2):
        x1 = self.flatten(feat_s1)
        x2 = self.flatten(feat_s2)
        x1 = self.relu(self.shared_fc_0(x1))
        x2 = self.relu(self.shared_fc_0(x2))

        # view 1
        cp1_s1 = self.En1(x1)
        cp1_s2 = self.En1(x2)
        re1_s1 = self.De1(cp1_s1)
        re1_s2 = self.De1(cp1_s2)

        cp1_s1_r = self.grl(cp1_s1)
        cp1_s2_r = self.grl(cp1_s2)
        cp1_s1_r = self.relu(self.shared_fc_1(cp1_s1_r))
        cp1_s2_r = self.relu(self.shared_fc_1(cp1_s2_r))

        dc1_s1 = self.dc1_scores(cp1_s1_r)
        dc1_s2 = self.dc1_scores(cp1_s2_r)
        labels_s1 = self.ones(dc1_s1.shape[0], ms.int32)
        labels_s2 = self.zeros(dc1_s2.shape[0], ms.int32)
        labels_s1 = self.onehot(labels_s1, self.num_classes, self.on_value, self.off_value)
        labels_s2 = self.onehot(labels_s2, self.num_classes, self.on_value, self.off_value)

        dc1_loss = self.loss(dc1_s1, labels_s1) + self.loss(dc1_s2, labels_s2)
        re1_loss = self.mse(re1_s1, x1) + self.mse(re1_s2, x2)

        # view 2
        cp2_s1 = self.En2(x1)
        cp2_s2 = self.En2(x2)
        re2_s1 = self.De2(cp2_s1)
        re2_s2 = self.De2(cp2_s2)
        
        cp2_s1_r = self.grl(cp2_s1)
        cp2_s2_r = self.grl(cp2_s2)
        cp2_s1_r = self.relu(self.shared_fc_1(cp2_s1_r))
        cp2_s2_r = self.relu(self.shared_fc_1(cp2_s2_r))

        dc2_s1 = self.dc2_scores(cp2_s1_r)
        dc2_s2 = self.dc2_scores(cp2_s2_r)

        dc2_loss = self.loss(dc2_s1, labels_s1) + self.loss(dc2_s2, labels_s2)
        re2_loss = self.mse(re2_s1, x1) + self.mse(re2_s2, x2)

        # view 3
        cp3_s1 = self.En3(x1)
        cp3_s2 = self.En3(x2)
        re3_s1 = self.De3(cp3_s1)
        re3_s2 = self.De3(cp3_s2)
        
        cp3_s1_r = self.grl(cp3_s1)
        cp3_s2_r = self.grl(cp3_s2)
        cp3_s1_r = self.relu(self.shared_fc_1(cp3_s1_r))
        cp3_s2_r = self.relu(self.shared_fc_1(cp3_s2_r))

        dc3_s1 = self.dc3_scores(cp3_s1_r)
        dc3_s2 = self.dc3_scores(cp3_s2_r)

        dc3_loss = self.loss(dc3_s1, labels_s1) + self.loss(dc3_s2, labels_s2)
        re3_loss = self.mse(re3_s1, x1) + self.mse(re3_s2, x2)

        # view difference
        dif12_s1 = self.mse(cp1_s1, cp2_s1)
        dif12_s2 = self.mse(cp1_s2, cp2_s2)
        dif23_s1 = self.mse(cp3_s1, cp2_s1)
        dif23_s2 = self.mse(cp3_s2, cp2_s2)
        dif13_s1 = self.mse(cp1_s1, cp3_s1)
        dif13_s2 = self.mse(cp1_s2, cp3_s2)
        loss_dif = dif12_s1 + dif12_s2 + dif23_s1 + dif23_s2 + dif13_s1 + dif13_s2
        
        loss_re = re1_loss + re2_loss + re3_loss
        loss_dc = dc1_loss + dc2_loss + dc3_loss

        loss = ops.exp(-loss_dif) + loss_re + loss_dc
        
        ins_MV_s1 = (dc1_s1 + dc2_s1 + dc3_s1)/3
        ins_MV_s2 = (dc1_s2 + dc2_s2 + dc3_s2)/3

        return loss, ins_MV_s1, ins_MV_s2

    def loss(self, cls_score, labels):
        """Loss method."""
        loss_dom, _ = self.loss_dom(cls_score, labels)

        return loss_dom

class Img_D(nn.Cell):
    def __init__(self, in_ch):
        super(Img_D, self).__init__()
        
        self.in_ch = in_ch

        self.ImgEn_1 = AutoEncoder(in_ch)
        self.ImgEn_2 = AutoEncoder(in_ch)
        self.ImgEn_3 = AutoEncoder(in_ch)
        self.ImgDe_1 = AutoDecoder(in_ch)
        self.ImgDe_2 = AutoDecoder(in_ch)
        self.ImgDe_3 = AutoDecoder(in_ch)

        self.ImgDC_1 = _ImageDA(128)    # 域判别头
        self.ImgDC_2 = _ImageDA(128)    # 域判别头
        self.ImgDC_3 = _ImageDA(128)    # 域判别头

        self.relu = ops.ReLU()
        self.loss_dom = ops.SoftmaxCrossEntropyWithLogits()
        self.ce_loss  =  nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.onehot = ops.OneHot()
        self.cast = ops.Cast()

        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.zeros = ops.Zeros()
        self.ones = ops.Ones()
        
        self.grl = GradReverse(0.2)
        self.mse = nn.MSELoss(reduction='mean')
        self.upsample = nn.ResizeBilinear()
        self.log_softmax = nn.LogSoftmax(axis=1)
        self.nnl = ops.NLLLoss()

    def construct(self, feat_s1, feat_s2):
        
        # 上采样
        feat_s1 = self.upsample(feat_s1, size=(191, 319))
        feat_s2 = self.upsample(feat_s2, size=(191, 319))
        
        # view 1
        cp1_s1 = self.ImgEn_1(feat_s1)
        cp1_s2 = self.ImgEn_1(feat_s2)
        re1_s1 = self.ImgDe_1(cp1_s1)
        re1_s2 = self.ImgDe_1(cp1_s2)

        MV1_score_s1 = self.ImgDC_1(cp1_s1)
        MV1_score_s2 = self.ImgDC_1(cp1_s2)
        MV1_label_s1 = self.ImageLabelResizeLayer(MV1_score_s1, 0.)
        MV1_label_s2 = self.ImageLabelResizeLayer(MV1_score_s2, 1.)

        # MV1_prob_s1_en = self.log_softmax(MV1_score_s1)
        DA_img_MV1_s1 = self.mse(MV1_score_s1, MV1_label_s1)

        # MV1_prob_s2_en = self.log_softmax(MV1_score_s2)
        DA_img_MV1_s2 = self.mse(MV1_score_s2, MV1_label_s2)

        dc1_loss = DA_img_MV1_s1 + DA_img_MV1_s2
        re1_loss = self.mse(re1_s1, feat_s1) + self.mse(re1_s2, feat_s2)

        # view 2
        cp2_s1 = self.ImgEn_2(feat_s1)
        cp2_s2 = self.ImgEn_2(feat_s2)
        re2_s1 = self.ImgDe_2(cp2_s1)
        re2_s2 = self.ImgDe_2(cp2_s2)
        
        MV2_score_s1 = self.ImgDC_2(cp2_s1)
        MV2_score_s2 = self.ImgDC_2(cp2_s2)
        MV2_label_s1 = self.ImageLabelResizeLayer(MV2_score_s1, 0.)
        MV2_label_s2 = self.ImageLabelResizeLayer(MV2_score_s2, 1.)

        # MV2_prob_s1_en = self.log_softmax(MV2_score_s1)
        DA_img_MV2_s1 = self.mse(MV2_score_s1, MV2_label_s1)

        # MV2_prob_s2_en = self.log_softmax(MV2_score_s2)
        DA_img_MV2_s2 = self.mse(MV2_score_s2, MV2_label_s2)

        dc2_loss = DA_img_MV2_s1 + DA_img_MV2_s2
        re2_loss = self.mse(re2_s1, feat_s1) + self.mse(re2_s2, feat_s2)

        # view 3
        cp3_s1 = self.ImgEn_3(feat_s1)
        cp3_s2 = self.ImgEn_3(feat_s2)
        re3_s1 = self.ImgDe_3(cp3_s1)
        re3_s2 = self.ImgDe_3(cp3_s2)
        
        MV3_score_s1 = self.ImgDC_3(cp3_s1)
        MV3_score_s2 = self.ImgDC_3(cp3_s2)
        MV3_label_s1 = self.ImageLabelResizeLayer(MV3_score_s1, 0.)
        MV3_label_s2 = self.ImageLabelResizeLayer(MV3_score_s2, 1.)

        # MV3_prob_s1_en = self.log_softmax(MV3_score_s1)
        DA_img_MV3_s1 = self.mse(MV3_score_s1, MV3_label_s1)

        # MV3_prob_s2_en = self.log_softmax(MV3_score_s2)
        DA_img_MV3_s2 = self.mse(MV3_score_s2, MV3_label_s2)

        dc3_loss = DA_img_MV3_s1 + DA_img_MV3_s2
        re3_loss = self.mse(re3_s1, feat_s1) + self.mse(re3_s2, feat_s2)

        # view difference
        dif12_s1 = self.mse(cp1_s1, cp2_s1)
        dif12_s2 = self.mse(cp1_s2, cp2_s2)
        dif23_s1 = self.mse(cp3_s1, cp2_s1)
        dif23_s2 = self.mse(cp3_s2, cp2_s2)
        dif13_s1 = self.mse(cp1_s1, cp3_s1)
        dif13_s2 = self.mse(cp1_s2, cp3_s2)
        loss_dif = dif12_s1 + dif12_s2 + dif23_s1 + dif23_s2 + dif13_s1 + dif13_s2

        loss_re = re1_loss + re2_loss + re3_loss
        
        loss_dc = dc1_loss + dc2_loss + dc3_loss
        # loss_dc = re1_loss # + dc2_loss + dc3_loss
        
        # loss = loss_re + loss_dif + dc1_loss
        loss = ops.exp(-loss_dif) + loss_re + loss_dc

        # soft cst
        img_MV_s1 = (MV1_score_s1 + MV2_score_s1 + MV3_score_s1)/3
        img_MV_s2 = (MV1_score_s2 + MV2_score_s2 + MV3_score_s2)/3

        return loss, img_MV_s1, img_MV_s2

    def loss(self, cls_score, labels):
        """Loss method."""
        loss_dom, _ = self.loss_dom(cls_score, labels)

        return loss_dom

        '''
    def ImageLabelResizeLayer(self, x, need_backprop):
        feats = x
        lbs = self.ones(x.shape[0], ms.float32) * need_backprop
        gt_blob = self.zeros((lbs.shape[0], 1, feats.shape[2], feats.shape[3]), ms.float32)
        for i in range(lbs.shape[0]):
            lb=lbs[i]
            # lbs_resize = cv2.resize(lb, (feats.shape[3] ,feats.shape[2]),  interpolation=cv2.INTER_NEAREST)
            lbs_resize = self.ones((feats.shape[2], feats.shape[3]), ms.float32) * lb
            # gt_blob[i, 0:lbs_resize.shape[0], 0:lbs_resize.shape[1], 0] = lbs_resize
            gt_blob[i, 0] = lbs_resize
            # y = xmj
            # lbs_resize = cv2.resize(1, (feats.shape[3] ,feats.shape[2]),  interpolation=cv2.INTER_NEAREST)

        # channel_swap = (0, 3, 1, 2)
        # gt_blob = gt_blob.transpose(channel_swap)
        # y = Tensor.from_numpy(gt_blob)
        gt_blob = ops.expand_dims(gt_blob, 1)
        # y=y.squeeze(1).long()
        return gt_blob
        '''

    def ImageLabelResizeLayer(self, feats, lb):
        gt_blob = self.ones((feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3]), ms.float32) * lb
        gt_blob[:,1,:,:] * (1.0 - lb)
        
        return gt_blob