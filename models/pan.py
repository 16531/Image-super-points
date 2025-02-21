import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import common


def compute_num_params(model, text=True):
    # tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    tot = sum([p.nelement() for p in model.parameters()])
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):
        super().__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)
        return out


class PAConv(nn.Module):
    def __init__(self, nf, k_size=3):
        super().__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution

    def forward(self, x):
        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out


class SCPA(nn.Module):
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super().__init__()
        group_width = nf // reduction

        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)

        self.k1 = nn.Sequential(
            nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                bias=False)
        )

        self.PAConv = PAConv(group_width)

        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual

        return out


class PAN(nn.Module):
    def __init__(self, nf=40, unf=24, nb=16, scale=4):
        super().__init__()
        in_nc = 3
        out_nc = 3

        self.scale = scale

        ### first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        ### main blocks
        m_SCPA_trunk = []
        for _ in range(nb):
            m_SCPA_trunk.append(SCPA(nf=nf, reduction=2))
        self.SCPA_trunk = nn.Sequential(*m_SCPA_trunk)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        #### upsampling
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        fea = fea + trunk

        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))

        out = self.conv_last(fea)

        ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + ILR
        return out


# if __name__ == '__main__':
#     x = torch.rand(1, 3, 48, 48)
#     model = PAN(scale=4)
#     y = model(x)
#     print(model)
#     print(y.shape)
#     print("param_nums:", compute_num_params(model, False))
