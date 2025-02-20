执行命令  python train.py --config=configs/.ipynb_checkpoints/elan_light_x4-checkpoint.yml

python train.py --config=configs/train_x4.yml

result中的carn，ctb，eca+unet等都是不同的对比方法  其中的子文件夹a是aid数据集 w是维护-rs19测试集
（carn是对比方法，a和w是不同的AID和WHU-RS19两个测试集的测试结果，）

执行命令   python test.py --config=configs/test_x4.yml   要用什么时候训练的模型,可以在.yml文件中改变训练文件的路径
从低清输出高清时，在test.py第91行改变输出路径，可以得到高清图片，要先建立文件夹，没有这个文件夹时候不会输出图片，但是可不影响程序运行
   


对于日志的解释:
##===========fp32-training, Epoch: 500, lr: [2.5e-05] =============##
[manga109-X4], PSNR/SSIM: 29.88/0.7989 (Best: 29.89/0.7993, Epoch: 451/480)
[b100-X4], PSNR/SSIM: 29.73/0.7858 (Best: 29.74/0.7862, Epoch: 451/451)
[u100-X4], PSNR/SSIM: 30.54/0.8095 (Best: 30.55/0.8097, Epoch: 380/417)
1.第一个是aid的验证集,第二个是aid的测试集,第三个是whu-rs19的测试集    在utils_train.py文件中配置



训练时的网络模型在此文件中models/elan_network_unet.py   
可以从train.py文件中的model = utils.import_module函数中知道，因为该函数在utils包中定义了
def import_module(name):
    return importlib.import_module(name)  #动态导入模块：import_module():返回指定的包或模块。
importlib是系统写好的包  可以查看它的用法从而知道它的用法


![image](https://github.com/user-attachments/assets/e43c15e2-3359-48cb-b74b-3565ccfe5d6e)



我的方法和其他方法对比
![{29F4DFA6-131F-46DF-81EC-E92582A6059D}](https://github.com/user-attachments/assets/a87a58d7-718e-4146-b011-b7f7fa53d49c)



模糊的图像

![image](https://github.com/user-attachments/assets/b78cb3f6-f963-4a70-9302-74e60872c88e)


经过我的网络还原成高清


![image](https://github.com/user-attachments/assets/04a19861-6426-4441-bc56-c55f866e6b81)
