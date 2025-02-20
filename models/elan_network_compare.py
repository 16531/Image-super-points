import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ops import MeanShift, UpsampleBlock

from models.vdsr import VDSR
from models.lgcnet import LGCNET
from models.pan import PAN
# from models.edsr import make_edsr
from models.carn import CARN_M


def create_model(args):
    return ELAN(args)


class ELAN(nn.Module):
    def __init__(self, args):
        super(ELAN, self).__init__()

        scale = 4  # value of scale is scale.
        multi_scale = 1  # value of multi_scale is multi_scale in args.
        group = 1  # if value of group isn't given, group is 1.
        kernel_size = 3  # tcw 201904091123
        padding = 1  # tcw201904091123
        features = 64  # tcw201904091124
        channels = 3
        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        '''
           in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
        '''
        # VDSR
        # self.b = VDSR(4).cuda()
        # LGCNet
        # self.b = LGCNET(scale=4)
        # EDSR
        # self.b = make_edsr(n_resblocks=32, n_feats=256, scale=4, rgb_range=3).cuda()
        # PAN
        # self.b = PAN(scale=4)
        # CARN_M
        self.b = CARN_M(scale=4)

    def forward(self, x):
        out = self.b(x)
        return out
