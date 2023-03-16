from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
import torch.nn as nn
from torch.autograd import Function
from model.faster_rcnn_MV_3.LabelResizeLayer import ImageLabelResizeLayer
from model.faster_rcnn_MV_3.LabelResizeLayer import InstanceLabelResizeLayer

class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)

class _ImageDA_res(nn.Module):
    def __init__(self,dim):
        super(_ImageDA_res,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1,bias=False)
        self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)
        self.LabelResizeLayer=ImageLabelResizeLayer()
        self.refineconv = nn.Sequential(Bottleneck(self.dim, self.dim // 4),
                                        Bottleneck(self.dim, self.dim // 4))
    def forward(self,x,need_backprop):
        x = grad_reverse(x)
        x = self.refineconv(x)
        x=self.reLu(self.Conv1(x))
        x=self.Conv2(x)
        label=self.LabelResizeLayer(x,need_backprop)
        return x,label


class _ImageDA(nn.Module):
    def __init__(self,dim):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1,bias=False)
        # self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)
        self.LabelResizeLayer=ImageLabelResizeLayer()

    def forward(self,x,need_backprop):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        x=self.Conv2(x)
        label=self.LabelResizeLayer(x,need_backprop)

        return x,label

class _ImageDA_noGRL(nn.Module):
    def __init__(self,dim):
        super(_ImageDA_noGRL,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1,bias=False)
        # self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)
        self.LabelResizeLayer=ImageLabelResizeLayer()

    def forward(self,x,need_backprop):
        x=self.reLu(self.Conv1(x))
        x=self.Conv2(x)
        label=self.LabelResizeLayer(x,need_backprop)

        return x,label

class _InstanceDA(nn.Module):
    def __init__(self):
        super(_InstanceDA,self).__init__()
        self.dc_ip1 = nn.Linear(4096, 1024)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(1024, 1024)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer=nn.Linear(1024,1)
        self.LabelResizeLayer=InstanceLabelResizeLayer()

    def forward(self,x,need_backprop):
        x=grad_reverse(x)
        x=self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x=self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x=F.sigmoid(self.clssifer(x))

        label = self.LabelResizeLayer(x, need_backprop)
        return x,label

class _InstanceDA_En(nn.Module):
    def __init__(self):
        super(_InstanceDA_En,self).__init__()
        self.dc_ip1 = nn.Linear(512, 256)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(256, 64)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer=nn.Linear(64, 1)
        self.LabelResizeLayer=InstanceLabelResizeLayer()

    def forward(self,x,need_backprop):
        x=grad_reverse(x)
        x=self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x=self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x=F.sigmoid(self.clssifer(x))

        label = self.LabelResizeLayer(x, need_backprop)
        return x,label


