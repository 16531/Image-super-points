#benchmark 基准测试#
import os
import glob  #glob是Python自带的一个文件操作相关的模块，可以查询符合自己要求的文件。支持通配符操作： ,?[] 这三个通配符
import random  #random模块用于生成随机数
import pickle  #Python标准库，只支持python的基本数据类型。可以处理复杂的序列化语法，序列化时，只是序列化整个序列对象，而非内存地址。
               #解释：加工数据使用，可以用来存取结构化数据。

import numpy as np
import imageio  #读取照片RGB内容，转换照片格式
import torch
import torch.utils.data as data
import skimage.color as sc  #skimage即是scikit-image，数字图片处理包。skimage.color是颜色空间变换。
from torch.utils.data import DataLoader  #主要是对数据进行batch的划分，输入进函数的数据一定是可迭代的。DataLoader的好处是可以快速迭代数据。
import time
import utils   #数据类型转换
import cv2


class Benchmark(data.Dataset):
    def __init__(self, HR_folder, LR_folder, scale=4, colors=1):  #__init__():用于类的初始化，负责创建类的实例属性并进行赋值等重要操作。
        super(Benchmark, self).__init__()
        self.HR_folder = HR_folder
        self.LR_folder = LR_folder

        self.img_postfix = '.jpg'
        self.scale = scale
        self.colors = colors

        self.nums_dataset = 0

        self.hr_filenames = []
        self.lr_filenames = []
        ## generate dataset  生成数据集
        tags = os.listdir(self.HR_folder)  #os.listdir()方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
        for tag in tags:
            hr_filename = os.path.join(self.HR_folder, tag)  #os.path.join：将多个路径组合后返回。
            lr_filename = os.path.join(self.LR_folder, tag)      #(self.LR_folder,'X{}'.format(scale), tag.replace('.png', 'x{}.png'.format(self.scale)))
            #str.format()函数：格式化字符串，通过{}和：来代替%。可以不限参数，位置不按顺序。
            #replace()方法：把字符串中的old(旧字符串)替换成new(新字符串)，如果指定第三个参数max，则替换不超过max次。语法：str.replace(old, new[, max])
            self.hr_filenames.append(hr_filename)  #append()函数：用来在列表末尾添加新的对象。语法：list.append(obj) list：列表对象；obj：添加到列表末尾的对象。
            self.lr_filenames.append(lr_filename)
        self.nums_trainset = len(self.hr_filenames)  #len(str):返回字符串、列表、字典、元组等长度。
        ## if store in ram
        self.hr_images = []
        self.lr_images = []

        LEN = len(self.hr_filenames)
        for i in range(LEN):
            lr_image, hr_image = imageio.imread(self.lr_filenames[i], pilmode="RGB"), imageio.imread(self.hr_filenames[i], pilmode="RGB")
            # lr_image, hr_image = cv2.imread(self.lr_filenames[i]), cv2.imread(self.hr_filenames[i])
            #imread():读取图片文件，以多维数组的形式保存图片信息，前两维表示图片的像素坐标，最后一维表示图片的通道索引。两个参数：图片路径和读取图片的形式。
            if self.colors == 1:
                lr_image, hr_image = sc.rgb2ycbcr(lr_image)[:, :, 0:1], sc.rgb2ycbcr(hr_image)[:, :, 0:1]  #RGB空间转YCbCr空间
                #YCbCr：YCBCR，Y是颜色的亮度成分，CB和CR则为蓝色和红色的浓度偏移量成分。子采样格式有4:2:0、 4：2：2、 4：4：4。
            self.hr_images.append(hr_image)
            self.lr_images.append(lr_image) 

    #自定义一个类，测量某个实例属性的长度
    def __len__(self):  #测量某个实例属性(self.hr_filenames)的长度。
        return len(self.hr_filenames)

    #自己定义一个类，使用索引的方式获取这个类实例的属性值。
    def __getitem__(self, idx):
        # get whole image, store in ram by default
        #__getitem__():接收一个idx参数，即索引值。当__getitem__()被触发后，就会返回lr，hr。
        lr, hr = self.lr_images[idx], self.hr_images[idx]
        lr_h, lr_w, _ = lr.shape  #shape：用于计算数组的行数列数。shape[0]代表行数，shape[1]代表列数，shape行和列构成的元组。
        hr = hr[0:lr_h*self.scale, 0:lr_w*self.scale, :]
        lr, hr = utils.ndarray2tensor(lr), utils.ndarray2tensor(hr)  #ndarray数据类型转tensor
        return lr, hr

if __name__ == '__main__':
    HR_folder = './SR_datasets/AID/val/HR'
    LR_folder = './SR_datasets/AID/val/LR'
    benchmark = Benchmark(HR_folder, LR_folder, scale=4, colors=1)
    benchmark = DataLoader(dataset=benchmark, batch_size=1, shuffle=False)
    #torch.utils.data.DataLoader的参数： dataset：定义的dataset类返回的结果；batchsize：每个batch要加载的样本数，默认为1；
    #shuffle：在每个epoch中对整个数据集data进行shuffle重排，默认为False；
    #sample：定义从数据集中加载数据所采用的策略；如果指定的话，shuffle必须为False；batch_sample类似，表示一次返回一个batch的index。
    #num_workers：表示开启多少个线程数去加载你的数据，默认为0，代表只使用主进程。
    #collate_fn：表示合并样本列表以形成小批量的Tensor对象。
    #pin_memory：表示要将load进来的数据是否要拷贝到pin_memory区中，其表示生成的Tensor数据是属于内存中的锁页内存区，这样将Tensor数据转义到GPU中速度就会快一些，默认为False。
    #drop_last：当你的整个数据长度不能够整除你的batchsize，选择是否要丢弃最后一个不完整的batch，默认为False。

    print("numner of sample: {}".format(len(benchmark.dataset)))  #str.format()函数：格式化字符串，通过{}和：来代替%。可以不限参数，位置不按顺序。
    start = time.time()  #获取当前时间
    for lr, hr in benchmark:
        print(lr.shape, hr.shape)
    end = time.time()
    print(end - start)  #进行基准测试花费的时间