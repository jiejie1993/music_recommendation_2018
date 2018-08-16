# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import numpy as np
import os
from surprise import Reader
from surprise import Dataset
import pandas as pd
#:pickle模块是对Python对象结构进行二进制序列化和反序列化的协议实现,就是把Python数据变成流的形式

def read_data_and_process(filname, sep="\t"):
#    col_names = ["user", "item", "rate", "st"]
#    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
#    df["user"] -= 1
#    df["item"] -= 1
#    for col in ("user", "item"):
#        df[col] = df[col].astype(np.int32)
#    df["rate"] = df["rate"].astype(np.float32)
#    return df


    file_path = os.path.expanduser(filname)
    # 指定文件格式
    reader = Reader(line_format='user item rating timestamp', sep=',')
    # 从文件读取数据
    music_data = Dataset.load_from_file(file_path, reader=reader)
    # 构建数据集和建模
    trainset = music_data.build_full_trainset()

    return trainset


class ShuffleDataIterator(object):
    """
    随机生成一个batch一个batch数据
    """
    #初始化
    def __init__(self, inputs, batch_size=10):
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))#向量进行拼接

    #总样本量
    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    #取出下一个batch
    def __next__(self):
        return self.next()

    #随机生成batch_size个下标，取出对应的样本
    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size,))
        out = self.inputs[ids, :]
        return [out[:, i] for i in range(self.num_cols)]


class OneEpochDataIterator(ShuffleDataIterator):
    """
    顺序产出一个epoch的数据，在测试中可能会用到
    """
    def __init__(self, inputs, batch_size=10):
        super(OneEpochDataIterator, self).__init__(inputs, batch_size=batch_size)
        if batch_size > 0:
            self.idx_group = np.array_split(np.arange(self.len), np.ceil(self.len / batch_size))
        else:
            self.idx_group = [np.arange(self.len)]
        self.group_id = 0

    def next(self):
        if self.group_id >= len(self.idx_group):
            self.group_id = 0
            raise StopIteration
        out = self.inputs[self.idx_group[self.group_id], :]
        self.group_id += 1
        return [out[:, i] for i in range(self.num_cols)]