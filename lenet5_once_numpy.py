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

#使用自己搭建的卷积和乘累加验证正确率单一测试

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

def write_file(fout,string,array):
    fout.write(">>> "+ string + ":\n")
    sh = np.shape(array)
    fout.write(str(np.shape(array)))
    fout.write("\n")
    #write data
    transfer = array.reshape(-1)
    temp = []
    for i in transfer:
        temp.append("%04x" % i)
    retransfer = np.array(temp)
    #retransfer = retransfer.reshape(sh)
    if len(sh) == 4: #如果是卷积结果
        for ch in range(sh[3]):
            fout.write("Channel %d:\n" % ch)
            for ver in range(sh[2]):
                fout.write("%02d:" % ver)
                for hor in range(sh[1]):
                    fout.write(" " + retransfer[ver*sh[2]*sh[3] +hor*sh[3] +ch])
                fout.write("\n")
    else:
        fout.write(str(retransfer))
        fout.write("\n\n")



fout = open('result.txt','w')

print("import data")
# Import data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

print("start predict")
save_file='./ckptlenet/last.ckpt'
reader=tf.train.NewCheckpointReader(save_file)
# Create the model
# 定义输入层
with open('input_origin784.txt','r') as fin:
    lines = fin.readlines()
xin = []
for line in lines:
    xin.append(int(line.rstrip(),16))
a= []
a.append(xin)
xin = np.array(a)
xin = xin.reshape(1,28,28,1).astype(np.int16)
ximg = xin.reshape(28,28)
#write_file(fout,'input',xin)

#plt.imshow(ximg,cmap='gray')
#plt.axis('off')
#plt.show()
#定义第一层卷积
#第1层：卷积 6  5x5 -> 28x28x6
W_conv1 = change_int(reader.get_tensor('Layer1/L1_weight'))
b_conv1 = change_int(reader.get_tensor('Layer1/L1_bias'))
conv1_mul = conv2d_same(xin, W_conv1)
conv1_add = conv_add(conv1_mul,b_conv1)
result_l1 = relu(conv1_add)
write_file(fout,'result Layer 1',result_l1)
#第2层：池化 2x2    -> 14x14x6
result_l2 = max_pool_2x2(result_l1)
write_file(fout,'result Layer 2',result_l2)
#第3层：卷积 16 5x5 -> 10x10x16
W_conv2 = change_int(reader.get_tensor('Layer3/L3_weight'))
b_conv2 = change_int(reader.get_tensor('Layer3/L3_bias'))
conv2_mul = conv2d(result_l2, W_conv2)
conv2_add = conv_add(conv2_mul,b_conv2)
result_l3 = relu(conv2_add)
write_file(fout,'result Layer 3',result_l3)
#第4层：池化 2x2    -> 5x5x16
result_l4 = max_pool_2x2(result_l3)
write_file(fout,'result Layer 4',result_l4)
#第5层：全连接 120  -> 120
result_l4_flat = result_l4.reshape([-1, 5*5*16])
W_fc1 = change_int(reader.get_tensor('Layer5/F1_weight'))
b_fc1 = change_int(reader.get_tensor('Layer5/F1_bias'))
fc1_mul = matmul(result_l4_flat, W_fc1)
fc1_result_relu = relu(fc1_mul + b_fc1)
result_l5 = fc1_result_relu
write_file(fout,'result Layer 5',result_l5)
#第6层：全连接 84   ->84
W_fc2 = change_int(reader.get_tensor('Layer6/F2_weight'))
b_fc2 = change_int(reader.get_tensor('Layer6/F2_bias'))
fc2_mul = matmul(result_l5, W_fc2)
fc2_result_relu = relu(fc2_mul + b_fc2)
result_l6 = fc2_result_relu
write_file(fout,'result Layer 6',result_l6)
#第7层：全连接 10   ->10
W_fc3 = change_int(reader.get_tensor('Layer7/F3_weight'))
b_fc3 = change_int(reader.get_tensor('Layer7/F3_bias'))
fc3_mul = matmul(result_l6, W_fc3)
result_l7 = relu(fc3_mul + b_fc3)
write_file(fout,'result Layer 7',result_l7)




fout.close()
