# 准备数据集图像hr和lr #
import os
import glob
import random
import pickle  # Python标准库，只支持python的基本数据类型。可以处理复杂的序列化语法，序列化时，只是序列化整个序列对象，而非内存地址。
               # 解释：加工数据使用，可以用来存取结构化数据。

import numpy as np
import imageio  #读取照片RGB内容，转换照片格式
import torch
import torch.utils.data as data
import skimage.color as sc
import time
from utils import ndarray2tensor


def crop_patch(lr, hr, patch_size, scale, augment=True):  # 定义函数crop_patch
    # crop patch randomly
    lr_h, lr_w, _ = lr.shape  # shape：用于计算数组的行数列数。shape[0]代表行数，shape[1]代表列数，shape行和列构成的元组。
    hp = patch_size
    lp = patch_size // scale
    lx, ly = random.randrange(0, lr_w - lp + 1), random.randrange(0, lr_h - lp + 1)
    # randrange():返回指定递增基数集合中的一个随机数，基数默认值为1。
    hx, hy = lx * scale, ly * scale
    lr_patch, hr_patch = lr[ly:ly+lp, lx:lx+lp, :], hr[hy:hy+hp, hx:hx+hp, :]
    # augment data扩充数据
    if augment:
        hflip = random.random() > 0.5  # 生成随机数
        vflip = random.random() > 0.5
        rot90 = random.random() > 0.5
        if hflip:
            lr_patch, hr_patch = lr_patch[:, ::-1, :], hr_patch[:, ::-1, :]
        if vflip:
            lr_patch, hr_patch = lr_patch[::-1, :, :], hr_patch[::-1, :, :]
        if rot90:
            lr_patch, hr_patch = lr_patch.transpose(1, 0, 2), hr_patch.transpose(1, 0, 2)
            # Pandas Series.transpose():返回转置，按定义是self。
        # numpy to tensor转换数据类型
    lr_patch, hr_patch = ndarray2tensor(lr_patch), ndarray2tensor(hr_patch)
    return lr_patch, hr_patch


class DIV2K(data.Dataset):  #定义类
    def __init__(
        self, HR_folder, LR_folder, CACHE_folder, 
        train=True, augment=True, scale=4, colors=1, 
        patch_size=96, repeat=168
    ):
        super(DIV2K, self).__init__()
        self.HR_folder = HR_folder
        self.LR_folder = LR_folder
        self.augment = augment
        self.img_postfix = '.jpg'
        self.scale = scale
        self.colors = colors
        self.patch_size = patch_size
        self.repeat = repeat
        self.nums_trainset = 0
        self.train = train
        self.cache_dir = CACHE_folder

        # for raw png images 真实的png图像
        self.hr_filenames = []
        self.lr_filenames = []
        # for numpy array data numpy数据
        self.hr_npy_names = []
        self.lr_npy_names = []
        # store in ram 存储在ram
        self.hr_images = []
        self.lr_images = []

        # generate dataset 生成数据集
        if self.train:
            self.start_idx = 0
            self.end_idx = 8029
        else:
            self.start_idx = 8031
            self.end_idx = 1647

        for i in range(self.start_idx, self.end_idx):
            idx = str(i).zfill(4)  # zfill()：返回指定长度的字符串，原字符串右对齐，前边填充0.语法：str.zfill(width).这里长度为4
            hr_filename = os.path.join(self.HR_folder + '/' + idx + self.img_postfix)  #os.path.join：将多个路径组合后返回。
            lr_filename = os.path.join(self.LR_folder + '/' + idx + self.img_postfix) 
            # lr_filename = os.path.join(self.LR_folder, 'X{}'.format(self.scale), idx + 'x{}'.format(self.scale) + self.img_postfix)
            self.hr_filenames.append(hr_filename)
            self.lr_filenames.append(lr_filename)
        self.nums_trainset = len(self.hr_filenames)

        LEN = self.end_idx - self.start_idx
        hr_dir = os.path.join(self.cache_dir, 'set_hr', 'ycbcr' if self.colors == 1 else 'rgb')
        lr_dir = os.path.join(self.cache_dir, 'set_lr_x{}'.format(self.scale), 'ycbcr' if self.colors == 1 else 'rgb')
        if not os.path.exists(hr_dir):  # os.path.exists():判断文件或文件夹是否存在
            os.makedirs(hr_dir)
            # 用于递归创建目录。如果子目录创建失败或已经存在，会抛出一个OSError的异常。
            # 语法：os.makedirs(path,mode) mode：权限模式
        else:
            for i in range(LEN):
                hr_npy_name = self.hr_filenames[i].split('/')[-1].replace('.jpg', '.npy')
                # split():拆分字符串。通过指定分隔符对字符串进行切片，并返回分割后的字符串列表(list) os.path.split():按照路径将文件名和路径分割开。
                # replace()方法：把字符串中的old(旧字符串)替换成new(新字符串)，如果指定第三个参数max，则替换不超过max次。语法：str.replace(old, new[, max])
                hr_npy_name = os.path.join(hr_dir, hr_npy_name)
                self.hr_npy_names.append(hr_npy_name)
            
        if not os.path.exists(lr_dir):
            os.makedirs(lr_dir)
        else:
            for i in range(LEN):
                lr_npy_name = self.lr_filenames[i].split('/')[-1].replace('.jpg', '.npy')
                lr_npy_name = os.path.join(lr_dir, lr_npy_name)
                self.lr_npy_names.append(lr_npy_name)

        # prepare hr images 准备hr图像
        if len(glob.glob(os.path.join(hr_dir, "*.npy"))) != len(self.hr_filenames):
            # glob.glob():返回所有匹配的文件路径列表。参数pathname定义了文件路径匹配规则。
            for i in range(LEN):
                if (i+1) % 1000 == 0:
                    print("convert {} hr images to npy data!".format(i+1))
                hr_image = imageio.imread(self.hr_filenames[i], pilmode="RGB")
                # imread():读取图片文件，以多维数组的形式保存图片信息，前两维表示图片的像素坐标，最后一维表示图片的通道索引。两个参数：图片路径和读取图片的形式。
                if self.colors == 1:
                    hr_image = sc.rgb2ycbcr(hr_image)[:, :, 0:1]  #转换颜色空间
                hr_npy_name = self.hr_filenames[i].split('/')[-1].replace('.jpg', '.npy')
                hr_npy_name = os.path.join(hr_dir, hr_npy_name)
                self.hr_npy_names.append(hr_npy_name)
                np.save(hr_npy_name, hr_image)
                # np.save(file,arr,allow_pickle=True,fix_imports=True):以".npy"格式将数组保存到二进制文件中。
                # file：要保存的文件名称，需指定文件保存路径；arr：要保存的数组。
        else:
            print("hr npy datas have already been prepared!, hr: {}".format(len(self.hr_npy_names)))  #hr图像已经准备好了
        # prepare lr images
        if len(glob.glob(os.path.join(lr_dir, "*.npy"))) != len(self.lr_filenames):
            for i in range(LEN):
                if (i+1) % 1000 == 0:
                    print("convert {} lr images to npy data!".format(i+1))
                lr_image = imageio.imread(self.lr_filenames[i], pilmode="RGB")
                if self.colors == 1:
                    lr_image = sc.rgb2ycbcr(lr_image)[:, :, 0:1]
                lr_npy_name = self.lr_filenames[i].split('/')[-1].replace('.jpg', '.npy')
                lr_npy_name = os.path.join(lr_dir, lr_npy_name)
                self.lr_npy_names.append(lr_npy_name)
                np.save(lr_npy_name, lr_image)
        else:
            print("lr npy datas have already been prepared!, lr: {}".format(len(self.lr_npy_names)))

    def __len__(self):
        if self.train:
            return self.nums_trainset * self.repeat
        else:
            return self.nums_trainset

    # 自己定义一个类，使用索引的方式获取这个类实例的属性值。
    def __getitem__(self, idx):
        # get periodic index 获取定期索引
        idx = idx % self.nums_trainset  # %：模运算，取余数
        # get whole image
        hr, lr = np.load(self.hr_npy_names[idx]), np.load(self.lr_npy_names[idx])  # np.load：读取数组数据，.npy文件
        if self.train:
            train_lr_patch, train_hr_patch = crop_patch(lr, hr, self.patch_size, self.scale, True)  # 调用定义的crop_patch函数
            return train_lr_patch, train_hr_patch
        return lr, hr

if __name__ == '__main__':
    HR_folder = 'E:/ruanjian/pycharm/program/elan/SR_datasets/AID/train/HR'
    LR_folder = 'E:/ruanjian/pycharm/program/elan/SR_datasets/AID/train/LR'
    # CACHE_folder = 'E:/ruanjian/pycharm/program/elan/SR_datasets/AID/train/aid_cache'
    augment = True  # 扩充
    div2k = DIV2K(HR_folder, LR_folder, augment=True, scale=4, colors=3, patch_size=96, repeat=1)

    print("numer of sample: {}".format(len(div2k)))
    start = time.time()
    for idx in range(10):
        tlr, thr, vlr, vhr = div2k[idx]
        print(tlr.shape, thr.shape, vlr.shape, vhr.shape)
    end = time.time()
    print(end - start)
