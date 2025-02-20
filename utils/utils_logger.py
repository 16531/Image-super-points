# Copyright (c) 2022 Yawei Li, Kai Zhang, Radu Timofte SPDX license identifier: MIT
import os
import sys
import datetime
#datetime模块有5个常用类：date、time、datetime、timedelta、tzinfo。
import logging  #logging定义了为应用程序和库实现灵活的事件日志记录的函数和类。


'''
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
https://github.com/xinntao/BasicSR
'''


def log(*args, **kwargs):  #log()：返回x的自然对数。
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


'''
# ===============================
# logger
# logger_name = None = 'base' ???
# ===============================
'''


def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger创建logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    log = logging.getLogger(logger_name)  #logger从来不直接实例化，经常通过logging模块级方法logging.getLogger来获得。如果name不给定就用root。
    #设置logging，创建一个FileHandler，并对输出消息的格式进行设置，将其添加到logger，然后将日志写入指定的文件中。
    if log.hasHandlers():  #handler名称：位置；作用
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        #%(asctime)s：打印日志的时间；%(message)s：打印日志信息；datefmt：指定时间的输出格式。
        fh = logging.FileHandler(log_path, mode='a')  #logging.FileHandler：日志输出到文件
        fh.setFormatter(formatter)  #通过setFormatter()方法设置了一个Formatter对象，因此输出的内容便是格式化后的日志信息。
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()  #logging.StreamHandler()：日志输出到流，可以是sys.stderr,sys.stdout或者文件
        sh.setFormatter(formatter)
        log.addHandler(sh)


'''
# ===============================
# print to file and std_out simultaneously
# ===============================
'''


class logger_print(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  # write the message

    def flush(self):  #flush()：用于刷新内部缓冲区，关闭后python会自动刷新文件。
        pass
