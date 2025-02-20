###### esrgcnn ######
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.esrgcnn import MFCModule
from models.ops import MeanShift, UpsampleBlock


def create_model(args):
    return ELAN(args)


class ELAN(nn.Module):
    def __init__(self, args):
        super(ELAN, self).__init__()

        scale = 4  # value of scale is scale.
        multi_scale = 1  # value of multi_scale is multi_scale in args.
        group = 1   # if valule of group isn't given, group is 1.
        kernel_size = 3   # tcw 201904091123
        padding = 1   # tcw201904091123
        features = 64  # tcw201904091124
        channels = 3
        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        '''
           in_channels, out_channels, kernel_size, stride, padding, dialation, groups,
        '''

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, groups=1,
                      bias=False), nn.ReLU(inplace=False))
        self.b1 = MFCModule(features, features)
        self.b2 = MFCModule(features, features)
        self.b3 = MFCModule(features, features)
        self.b4 = MFCModule(features, features)
        self.b5 = MFCModule(features, features)
        self.b6 = MFCModule(features, features)

        # self.b1 = MFCModule(features)
        # self.b2 = MFCModule(features)
        # self.b3 = MFCModule(features)
        # self.b4 = MFCModule(features)
        # self.b5 = MFCModule(features)
        # self.b6 = MFCModule(features)
        # self.b7 = MFCModule(features)
        # self.b8 = MFCModule(features)
        self.ReLU = nn.ReLU(inplace=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=1,
                      bias=False), nn.ReLU(inplace=False))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=3, kernel_size=kernel_size, padding=padding, groups=1,
                      bias=False))
        self.upsample = UpsampleBlock(features, scale=scale, multi_scale=multi_scale, group=1)

    def forward(self, x):
        x = self.sub_mean(x)
        x1 = self.conv1_1(x)

        b1 = self.b1(x1)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        b5 = self.b5(b4)
        b6 = self.b6(b5)
        b7 = self.b6(b6)
        b8 = self.b6(b7)

        x2 = self.conv2(b8)
        temp = self.upsample(x2, scale=4)
        temp2 = self.ReLU(temp)
        out = self.conv3(temp2)
        out = self.add_mean(out)
        return out
