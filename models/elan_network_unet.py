import torch
import torch.nn as nn
from models.ops import MeanShift, UpsampleBlock
from models.ECA_unet import MFCModule
from models.unet import UNet


def create_model(args):
    return ELAN(args)


class DConv3x3(nn.Module):  # 经过一个深度可分离改变通道，由输出通道数决定
    def __init__(self, in_channel, out_channel):
        # 这一行千万不要忘记
        super(DConv3x3, self).__init__()
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class DConv5x5(nn.Module):  # 经过一个深度可分离改变通道，由输出通道数决定
    def __init__(self, in_channel, out_channel):
        # 这一行千万不要忘记
        super(DConv5x5, self).__init__()
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=5,
                                    stride=1,
                                    padding=2,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class DConv7x7(nn.Module):  # 经过一个深度可分离改变通道，由输出通道数决定
    def __init__(self, in_channel, out_channel):
        # 这一行千万不要忘记
        super(DConv7x7, self).__init__()
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=7,
                                    stride=1,
                                    padding=3,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class DConv(nn.Module):
    def __init__(self, inc, ouc):
        super(DConv, self).__init__()
        self.conv1 = DConv3x3(inc, ouc)
        self.conv2 = DConv5x5(inc, ouc)
        self.conv3 = DConv7x7(inc, ouc)
        self.conv = nn.Conv2d(in_channels=ouc, out_channels=ouc, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv(x1)

        y1 = self.conv2(x)
        y2 = self.conv(y1)

        z1 = self.conv3(x)
        z2 = self.conv(z1)

        out = torch.cat([x2, y2, z2], dim=1)
        return out


class ELAN(nn.Module):
    def __init__(self, args):
        super(ELAN, self).__init__()

        scale = 4  # value of scale is scale.
        multi_scale = 1  # value of multi_scale is multi_scale in args.
        group = 1  # if valule of group isn't given, group is 1.
        kernel_size = 3  # tcw 201904091123
        padding = 1  # tcw201904091123
        features = 64   #96  # tcw201904091124
        channels = 3
        inp = 32
        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        '''
           in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
        '''

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, groups=1,
                      bias=False), nn.ReLU(inplace=False))
        # unet
        self.b1 = MFCModule(features, features)
        self.b2 = MFCModule(features, features)
        self.b3 = MFCModule(features, features)
        self.b4 = MFCModule(features, features)
        self.b5 = MFCModule(features, features)
        self.b6 = MFCModule(features, features)
        # aim_ct
        # self.b1 = MFCModule(features)
        # self.b2 = MFCModule(features)
        # self.b3 = MFCModule(features)
        # self.b4 = MFCModule(features)
        # self.b5 = MFCModule(features)
        # self.b6 = MFCModule(features)
        self.ReLU = nn.ReLU(inplace=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=1,
                      bias=False), nn.ReLU(inplace=False))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=3, kernel_size=kernel_size, padding=padding, groups=1,
                      bias=False))
        self.upsample = UpsampleBlock(features, scale=scale, multi_scale=multi_scale, group=1)
        self.unet = UNet(3, features)

        # self.conv = DConv(3, inp)

    def forward(self, x):
        x = self.sub_mean(x)
        x1 = self.conv1_1(x)
        # 加上unet
        x2 = self.unet(x)
        # 加上深度可分离卷积
        # x2 = self.conv(x)

        b1 = self.b1(x1 + x2)
        b2 = self.b2(b1 + x2)
        b3 = self.b3(b2 + x2)
        b4 = self.b4(b3 + x2)
        b5 = self.b5(b4 + x2)
        b6 = self.b6(b5 + x2)

        x2 = self.conv2(b6)
        temp = self.upsample(x2, scale=4)
        # temp = self.upsample(b6, scale=scale)
        temp2 = self.ReLU(temp)
        out = self.conv3(temp2)
        out = self.add_mean(out)
        return out
