import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ops import EResidualBlock, BasicBlock


class Block(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
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

    def forward(self, input):
        out1_c = self.conv1_1(input)
        dit1, remain1 = torch.split(out1_c, [self.distilled_channels, self.remaining_channels], dim=1)
        out1_r = self.ReLU(remain1)
        out2_c = self.conv1_2(out1_r)
        # out2_c = out2_c + out1_c
        dit2, remain2 = torch.split(out2_c, [self.distilled_channels, self.remaining_channels], dim=1)
        remain2 = remain2 + remain1
        out2_r = self.ReLU(remain2)
        out3_c = self.conv1_3(out2_r)
        # out3_c = out3_c + out2_c
        dit3, remain3 = torch.split(out3_c, [self.distilled_channels, self.remaining_channels], dim=1)
        remain3 = remain3 + remain2
        out3_r = self.ReLU(remain3)
        out4_c = self.conv1_4(out3_r)
        # out4_c = out4_c + out3_c
        dit4, remain4 = torch.split(out4_c, [self.distilled_channels, self.remaining_channels], dim=1)
        remain4 = remain4 + remain3
        dit = dit1 + dit2 + dit3 + dit4
        out_t = torch.cat([dit, remain4], dim=1)
        # out_t =  out1_c+out2_c+out3_c+out4_c+out_t
        out_r = self.ReLU(out_t)
        out5_r = self.conv1_5(out_r)
        out6_r = self.conv1_6(out5_r)
        out6_r = input + out6_r
        return out6_r



