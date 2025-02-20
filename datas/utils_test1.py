###test###
import os
from datas.benchmark import Benchmark
from datas.div2k import DIV2K
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import random
import numpy as np
from torchvision import transforms


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist


def create_datasets(args): #定义函数，构造数据

    valid_dataloaders = []
    if 'AID' in args.eval_sets:
        aid_hr_path = os.path.join(args.data_path, 'WHU-test/four/a/HR')
        aid_lr_path = os.path.join(args.data_path, 'WHU-test/four/a/LR')
        aid = Benchmark(aid_hr_path, aid_lr_path, scale=args.scale, colors=args.colors)
        valid_dataloaders += [{'name': 'aid', 'dataloader': DataLoader(dataset=aid, batch_size=1, shuffle=False)}]
    if 'WHU-RS19' in args.eval_sets:
        whu_hr_path = os.path.join(args.data_path, 'WHU-test/four/HR')
        whu_lr_path = os.path.join(args.data_path, 'WHU-test/four/LR')
        whu = Benchmark(whu_hr_path, whu_lr_path, scale=args.scale, colors=args.colors)
        valid_dataloaders += [{'name': 'whu', 'dataloader': DataLoader(dataset=whu, batch_size=1, shuffle=False)}]
    # 如果验证数据为0，则说明没有上述可选择的数据集
    if len(valid_dataloaders) == 0:
        print('select no dataset for evaluation!')
    else:
        selected = ''
        for i in range(1, len(valid_dataloaders)):
            selected += ", " + valid_dataloaders[i]['name']
        print('select {} for evaluation! '.format(selected))
    return valid_dataloaders
####test#####
