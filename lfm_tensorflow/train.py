# -*- coding: utf-8 -*-

import time
from collections import deque

import numpy as np
import tensorflow as tf
from six import next
from data_process import read_data_and_process
from tensorflow.core.framework import summary_pb2
import data_process
import lfm

np.random.seed(13575)

# 一批数据的大小
BATCH_SIZE = 2000
# 用户数
USER_NUM = 6040
# 电影数
ITEM_NUM = 3952
# factor维度
DIM = 15
# 最大迭代轮数
EPOCH_MAX = 200
# 使用cpu做训练
DEVICE = "/cpu:0"

# 截断
def clip(x):
    return np.clip(x, 1.0, 5.0)

# 这个是方便Tensorboard可视化做的summary
def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

# 调用函数获取数据
def get_data():
    df = read_data_and_process("./popular_music_suprise_format.txt", sep="::")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    print(df_train.shape, df_test.shape)
    return df_train, df_test

# 实际训练过程
def svd(train, test):
    samples_per_batch = len(train) // BATCH_SIZE

    # 一批一批数据用于训练
    iter_train = data_process.ShuffleDataIterator([train["user"],
                                         train["item"],
                                         train["rate"]],
                                        batch_size=BATCH_SIZE)
    # 测试数据
    iter_test = data_process.OneEpochDataIterator([test["user"],
                                         test["item"],
                                         test["rate"]],
                                        batch_size=-1)
    # user和item batch
    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])#是对应评分的batch

    # 构建graph和训练
    infer, regularizer = lfm.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
                                           device=DEVICE)  #infer是计算结果y（pred）  regulararizer是正则化部分
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op =lfm.optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.05, device=DEVICE)#这里表示计算图的cost以及迭代优化器

    # 初始化所有变量
    init_op = tf.global_variables_initializer()
    # 开始迭代
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(EPOCH_MAX * samples_per_batch):
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users, item_batch: items,rate_batch: rates})
           
            

 
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))#训练集误差 mse
                test_err2 = np.array([])   #测试集误差
                for users, items, rates in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,item_batch: items})
                    pred_batch = clip(pred_batch)
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, test_err,end - start))
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, test_err,end - start))
                train_err_summary = make_scalar_summary("training_error", train_err)
                test_err_summary = make_scalar_summary("test_error", test_err)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end
           
           
            #进行训练，包括mini_batch反向迭代更新参数,已经更新到输入矩阵中
            #fetches: A single graph element, or a list of graph elements (described above).
#feed_dict: A dictionary that maps graph elements to values (described above).

'''
This method runs one "step" of TensorFlow computation,
by running the necessary graph fragment to execute every Operation and evaluate every Tensor in fetches,
substituting the values in feed_dict for the corresponding input values.

Returns:
Either a single value if fetches is a single graph element, or a list of values if fetches is a list (described above).
'''
 

if __name__ == '__main__':
    # 获取数据
    df_train, df_test = get_data()

    # 完成实际的训练
    svd(df_train, df_test)