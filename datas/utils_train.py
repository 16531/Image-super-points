#  ###train###
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
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm',
                                          '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist


def create_datasets(args): #定义函数，构造数据
    # aid = DIV2K(
    #     os.path.join(args.data_path, 'Set5/HR'),  ##os.path.join：将多个路径组合后返回。
    #     os.path.join(args.data_path, 'Set5/LR'),
    #     os.path.join(args.data_path, 'set_cache'),
    #     train=True,
    #     augment=args.data_augment,
    #     scale=args.scale,
    #     colors=args.colors,
    #     patch_size=args.patch_size,
    #     repeat=args.data_repeat,
    # )
    # train_dataloader = DataLoader(dataset=aid, num_workers=args.threads, batch_size=args.batch_size, shuffle=True,
    #                               pin_memory=True, drop_last=True)
    aid = DIV2K(
        os.path.join(args.data_path, 'AID/train/HR'),  ##os.path.join：将多个路径组合后返回。
        os.path.join(args.data_path, 'AID/train/LR'),
        os.path.join(args.data_path, 'aid_cache'),
        train=True,
        augment=args.data_augment,
        scale=args.scale,
        colors=args.colors,
        patch_size=args.patch_size,
        repeat=args.data_repeat,
    )
    train_dataloader = DataLoader(dataset=aid, num_workers=args.threads, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, drop_last=True)

    valid_dataloaders = []
    if 'Manga109' in args.eval_sets:
        manga_hr_path = os.path.join(args.data_path, 'AID/val/HR')
        manga_lr_path = os.path.join(args.data_path, 'AID/val/LR')
        manga = Benchmark(manga_hr_path, manga_lr_path, scale=args.scale, colors=args.colors)
        valid_dataloaders += [{'name': 'manga109', 'dataloader': DataLoader(dataset=manga, batch_size=1, shuffle=False)}]
    # if 'B100' in args.eval_sets:
    #     b100_hr_path = os.path.join(args.data_path, 'AID/test/HR')
    #     b100_lr_path = os.path.join(args.data_path, 'AID/test/LR')
    #     b100 = Benchmark(b100_hr_path, b100_lr_path, scale=args.scale, colors=args.colors)
    #     valid_dataloaders += [{'name': 'b100', 'dataloader': DataLoader(dataset=b100, batch_size=1, shuffle=False)}]
    # if 'Urban100' in args.eval_sets:
    #     u100_hr_path = os.path.join(args.data_path, 'WHU-RS19/test/HR')
    #     u100_lr_path = os.path.join(args.data_path, 'WHU-RS19/test/LR')
    #     u100 = Benchmark(u100_hr_path, u100_lr_path, scale=args.scale, colors=args.colors)
    #     valid_dataloaders += [{'name': 'u100', 'dataloader': DataLoader(dataset=u100, batch_size=1, shuffle=False)}]
    # 如果验证数据为0，则说明没有上述可选择的数据集
    if len(valid_dataloaders) == 0:
        print('select no dataset for evaluation!')
    else:
        selected = ''
        for i in range(1, len(valid_dataloaders)):
            selected += ", " + valid_dataloaders[i]['name']
        print('select {} for evaluation! '.format(selected))
    return train_dataloader, valid_dataloaders
