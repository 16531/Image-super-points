import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.ops import EResidualBlock, BasicBlock
from torch.nn import init
from models.ACNet import ACBlock
from utils import trunc_normal_


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()
        self.b1 = EResidualBlock(64, 64, group=group)
        self.c1 = BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = BasicBlock(64 * 4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        return o3


class eca_layer(nn.Module):
    def __init__(self, channel, kernel_size):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return y


class ECAAttention(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        self.ac1 = ACBlock(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.ac2 = ACBlock(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        sp = 1 / 2
        self.c1 = int(inp * sp)
        self.c2 = int(inp - self.c1)

        self.eca = eca_layer(64, kernel_size=3)
        self.conv_s = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=inp, out_channels=inp, padding=2, dilation=2),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=False)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()  # x=[8,45,16,16]
        v = self.eca(x)
        k1, k2 = torch.split(x, [self.c1, self.c2], dim=1)
        k1 = self.ac1(k1)
        k2 = self.ac2(k2)
        k = torch.cat([k1, k2], dim=1)
        k = self.conv_s(k)
        out = k * v
        return out


class MFCModule(nn.Module):
    def __init__(self, in_channels, out_channels, gropus=1):
        super(MFCModule, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        features1 = 48
        distill_rate = 0.25
        self.distilled_channels = int(features * distill_rate)
        self.remaining_channels = int(features - self.distilled_channels)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=1,
                      bias=False))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features1, out_channels=features, kernel_size=kernel_size, padding=padding, groups=1,
                      bias=False))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features1, out_channels=features, kernel_size=kernel_size, padding=padding, groups=1,
                      bias=False))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features1, out_channels=features, kernel_size=kernel_size, padding=padding, groups=1,
                      bias=False))
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=1,
                      bias=False), nn.ReLU(inplace=False))
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=1, bias=False),
            nn.ReLU(inplace=False))
        self.ReLU = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1)
        # eca网络：
        # self.att1 = eca_layer(64, 3)
        # ECA网络：
        self.att1 = ECAAttention(64, 64)
        # ghost网络
        # self.att2 = att_block1(16, 16)
        # self.conv = GhostModule(inp=45, oup=60)
        # SKA网络
        # self.att1 = SKAAttention(64, 64)

    def forward(self, x):
        out1_c = self.conv1_1(x)
        dit1, remain1 = torch.split(out1_c, [self.distilled_channels, self.remaining_channels], dim=1)
        out1_r = self.ReLU(remain1)
        out2_c = self.conv1_2(out1_r)
        out2_c = self.att1(out2_c)
        dit1 = self.conv(dit1)

        # out2_c = out2_c + out1_c
        dit2, remain2 = torch.split(out2_c, [self.distilled_channels, self.remaining_channels], dim=1)
        remain2 = remain2 + remain1
        out2_r = self.ReLU(remain2)
        out3_c = self.conv1_3(out2_r)
        out3_c = self.att1(out3_c)
        dit2 = self.conv(dit2)

        # out3_c = out3_c + out2_c
        dit3, remain3 = torch.split(out3_c, [self.distilled_channels, self.remaining_channels], dim=1)
        remain3 = remain3 + remain2
        out3_r = self.ReLU(remain3)
        out4_c = self.conv1_4(out3_r)
        out4_c = self.att1(out4_c)
        dit3 = self.conv(dit3)

        # out4_c = out4_c + out3_c
        dit4, remain4 = torch.split(out4_c, [self.distilled_channels, self.remaining_channels], dim=1)
        dit4 = self.conv(dit4)
        remain4 = remain4 + remain3

        dit = dit1 + dit2 + dit3 + dit4
        out_t = torch.cat([dit, remain4], dim=1)
        # out_t =  out1_c+out2_c+out3_c+out4_c+out_t
        out_r = self.ReLU(out_t)
        out5_r = self.conv1_5(out_r)
        out6_r = self.conv1_6(out5_r)
        out6_r = x + out6_r
        return out6_r


