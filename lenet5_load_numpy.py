from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from lib_round import *

FLAGS = None
save_file='./ckptlenet_mul/last.ckpt'

#使用自己搭建的卷积和乘累加验证正确率，使用全部测试集

#输入：32x32
#第1层：卷积 6  5x5 -> 28x28x6
#第2层：池化 2x2    -> 14x14x6
#第3层：卷积 16 5x5 -> 10x10x16
#第4层：池化 2x2    -> 5x5x16
#第5层：全连接 120  -> 120
#第6层：全连接 84   ->84
#第7层：全连接 10   ->10

#6位整数，10位小数

#定义一堆函数

def change_int16(array):
    x1024 = array * 1024
    return x1024.astype(np.int16)


fout = open('result.txt','w')

print("import data")
# Import data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

print("start predict")
save_file='./ckptlenet/last.ckpt'
reader=tf.train.NewCheckpointReader(save_file)
# Create the model
# 定义输入层
imgs = mnist.test.next_batch(10000)
xmul = imgs[0] * 1024
xin  = xmul.astype(np.int16)
xin  = xin.reshape([-1,28,28,1])
print(xin.shape)
#定义第一层卷积
#第1层：卷积 6  5x5 -> 28x28x6
W_conv1 = change_int16(reader.get_tensor('Layer1/L1_weight'))
b_conv1 = change_int16(reader.get_tensor('Layer1/L1_bias'))
conv1_mul = conv2d_same(xin, W_conv1)
conv1_add = conv_add(conv1_mul,b_conv1)
result_l1 = relu(conv1_add)
print('layer 1 done')
#第2层：池化 2x2    -> 14x14x6
result_l2 = max_pool_2x2(result_l1)
print('layer 2 done')
#第3层：卷积 16 5x5 -> 10x10x16
W_conv2 = change_int16(reader.get_tensor('Layer3/L3_weight'))
b_conv2 = change_int16(reader.get_tensor('Layer3/L3_bias'))
conv2_mul = conv2d(result_l2, W_conv2)
conv2_add = conv_add(conv2_mul,b_conv2)
result_l3 = relu(conv2_add)
print('layer 3 done')
#第4层：池化 2x2    -> 5x5x16
result_l4 = max_pool_2x2(result_l3)
print('layer 4 done')
#第5层：全连接 120  -> 120
result_l4_flat = result_l4.reshape([-1, 5*5*16])
W_fc1 = change_int16(reader.get_tensor('Layer5/F1_weight'))
b_fc1 = change_int16(reader.get_tensor('Layer5/F1_bias'))
fc1_mul = matmul(result_l4_flat, W_fc1)
fc1_result_relu = relu(fc1_mul + b_fc1)
result_l5 = fc1_result_relu
print('layer 5 done')
#第6层：全连接 84   ->84
W_fc2 = change_int16(reader.get_tensor('Layer6/F2_weight'))
b_fc2 = change_int16(reader.get_tensor('Layer6/F2_bias'))
fc2_mul = matmul(result_l5, W_fc2)
fc2_result_relu = relu(fc2_mul + b_fc2)
result_l6 = fc2_result_relu
print('layer 6 done')
#第7层：全连接 10   ->10
W_fc3 = change_int16(reader.get_tensor('Layer7/F3_weight'))
b_fc3 = change_int16(reader.get_tensor('Layer7/F3_bias'))
fc3_mul = matmul(result_l6, W_fc3)
result_l7 = relu(fc3_mul + b_fc3)
print('layer 7 done')

y_conv = result_l7

y_ = imgs[1]

y_conv_argmax = np.argmax(y_conv, 1)
y_argmax      =  np.argmax(y_, 1)
correct_prediction = np.equal(y_conv_argmax,y_argmax)
#print(y_conv_argmax)
#print(y_argmax)
accuracy = correct_prediction.astype(np.float32).mean()
print('test accuracy %g' % accuracy)



fout.close()
