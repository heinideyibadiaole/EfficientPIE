"""

@ Description:
@ Project:APCIL
@ Author:qufang
@ Create:2024/6/19 01:09

"""
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
from functools import partial
from models.common import DropPath, SqueezeExcite, ConvBNAct, FusedMBConv, MBConv, AdaptiveEncoder


class EfficientPIE(nn.Module):
    def __init__(self,
                 num_classes: int = 2):
        super(EfficientPIE, self).__init__()
        # channel first
        # self.adaptive_encoder = AdaptiveEncoder(in_channels=24, out_channels=3)
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
        self.commonConv = ConvBNAct(3, 32, kernel_size=3, stride=2, norm_layer=norm_layer)
        self.fm1 = FusedMBConv(kernel_size=3, input_c=32, out_c=32, expand_ratio=1,
                               stride=1, se_ratio=0, drop_rate=0, norm_layer=norm_layer)
        self.fm2 = FusedMBConv(kernel_size=3, input_c=32, out_c=64, expand_ratio=4,
                               stride=2, se_ratio=0, drop_rate=0, norm_layer=norm_layer)
        self.mb1 = MBConv(kernel_size=3, input_c=64, out_c=128, expand_ratio=4,
                          stride=2, se_ratio=0.25, drop_rate=0, norm_layer=norm_layer)
        self.mb2 = MBConv(kernel_size=3, input_c=128, out_c=256, expand_ratio=4,
                          stride=2, se_ratio=0.25, drop_rate=0, norm_layer=norm_layer)
        self.commonConv1 = ConvBNAct(256, 1280, kernel_size=1, stride=1, norm_layer=norm_layer)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x = self.adaptive_encoder(x)
        x = self.commonConv(x)
        x = self.fm1(x)
        x = self.fm2(x)
        x = self.mb1(x)
        x = self.mb2(x)
        x = self.commonConv1(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        y = self.classifier(x)
        return y
