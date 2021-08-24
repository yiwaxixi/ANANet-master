import torch
import warnings
warnings.filterwarnings(action='ignore')
from module.mynet_ANANet import MyNet as ANANet
from torchsummary import summary

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#打印模型信息
model = ANANet(num_classes=21, backbone="resnet50", downsample_factor=16, aux_branch=True, pretrained=False).train().cuda()

#使用信息summary(pytorch_modele,input_size=(channels,H,W))

summary (model,(3,480,480))
