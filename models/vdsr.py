import torch
import torch.nn as nn
import torch.nn.functional as F
import time


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


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    def __init__(self, scale=2, n_resblocks=18):
        super().__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, n_resblocks)
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.scale = scale

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
        residual = x
        x = self.relu(self.input(x))
        out = self.residual_layer(x)
        # print("2", out.shape)
        out = self.output(out)
        out = torch.add(out, residual)
        return out


# # if __name__ == '__main__':
#     x = torch.rand(1, 3, 150, 150).cuda()
#     model = VDSR(4).cuda()
#     t = time.time()
#     y = model(x)
#     # print(model)
#     print("time ", time.time()-t)
#     param_nums = compute_num_params(model, True)
#     print(param_nums)
#     print(y.shape)

