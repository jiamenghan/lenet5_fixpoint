from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None
save_file='./ckptlenet/last.ckpt'

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

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
def conv2d_same(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    # 2x2的池化
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1] ,
            strides=[1, 2, 2, 1], padding='VALID')


def weight_variable(shape,name):
    """weight_variable generates a weight variable of a given shape."""
    # 产生一个初始化的正态分布的权重张量，均值是0，标准差是0.1，而且权重不会大于2倍的标准差
    initial = tf.truncated_normal(shape,dtype=tf.float32, stddev=0.1)
    # 返回图变量，该图变量具有初始值initial
    return tf.Variable(initial,name=name)


def bias_variable(shape,name):
    """bias_variable generates a bias variable of a given shape."""
    # 所有的偏移均为0.1
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial,name=name)


print("import data")
# Import data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

# Create the model
# 定义输入层

# Define loss and optimizer
# 定义输出结果
with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784],name='x_input')
    y_ = tf.placeholder(tf.float32, [None, 10],name='y_output')
    keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope('converse'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
#定义第一层卷积
#第1层：卷积 6  5x5 -> 28x28x6
with tf.variable_scope('Layer1'):
    W_conv1 = weight_variable([5, 5, 1, 6],'L1_weight')
    b_conv1 = bias_variable([6],'L1_bias')
    conv1_result = conv2d_same(x_image, W_conv1)
    result_l1 = tf.nn.relu(conv1_result + b_conv1)
#第2层：池化 2x2    -> 14x14x6
with tf.variable_scope('Layer2'):
        result_l2 = max_pool_2x2(result_l1)
#第3层：卷积 16 5x5 -> 10x10x16
with tf.variable_scope('Layer3'):
    W_conv2 = weight_variable([5, 5, 6, 16],'L3_weight')
    b_conv2 = bias_variable([16],'L3_bias')
    conv2_result = conv2d(result_l2, W_conv2)
    result_l3 = tf.nn.relu(conv2_result + b_conv2)
#第4层：池化 2x2    -> 5x5x16
with tf.variable_scope('Layer4'):
        result_l4 = max_pool_2x2(result_l3)
#第5层：全连接 120  -> 120
result_l4_flat = tf.reshape(result_l4, [-1, 5*5*16])
with tf.variable_scope('Layer5'):
    W_fc1 = weight_variable([5 * 5 * 16, 120],'F1_weight')
    b_fc1 = bias_variable([120],'F1_bias')
    fc1_result = tf.matmul(result_l4_flat, W_fc1)
    fc1_result_relu = tf.nn.relu(fc1_result + b_fc1)
    result_l5 = tf.nn.dropout(fc1_result_relu,keep_prob)
#第6层：全连接 84   ->84
with tf.variable_scope('Layer6'):
    W_fc2 = weight_variable([120, 84],'F2_weight')
    b_fc2 = bias_variable([84],'F2_bias')
    fc2_result = tf.matmul(result_l5, W_fc2)
    fc2_result_relu = tf.nn.relu(fc2_result + b_fc2)
    result_l6 = tf.nn.dropout(fc2_result_relu,keep_prob)
#第7层：全连接 10   ->10
with tf.variable_scope('Layer7'):
    W_fc3 = weight_variable([84, 10],'F3_weight')
    b_fc3 = bias_variable([10],'F3_bias')
    fc3_result = tf.matmul(result_l6, W_fc3)
    result_l7 = tf.nn.relu(fc3_result + b_fc3)

y_conv = result_l7

# 定义每次训练的损失函数和步长
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#cross_entropy_int16 = tf.cast(cross_entropy*1024,tf.int16)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 正确率判断，同样是搭建环境
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver=tf.train.Saver()

fout = open("predict.txt",'w')
print("start train")
with tf.Session() as sess:
    # 初始化所有的占位符，之前的函数叫initialize_all_variables，已被下式替代
    sess.run(tf.global_variables_initializer())
    #batch = mnist.train.next_batch(1)
    #print(y_conv.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
    for i in range(30000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            # 每次进行正确性测试，仅使用前50张图片
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1],keep_prob:1})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            fout.write("step %d, training accuracy %g\n" % (i, train_accuracy))
        # keep_prob 留存率，即dropout时需要训练的节点
        train_step.run(feed_dict={x: batch[0], y_: batch[1],keep_prob:0.5})

    # 最后的正确性测试，使用全部的图片
    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_prob:1}))
    fout.write('test accuracy %g\n' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_prob:1}))

    # 保存最后的权重
    saver.save(sess,save_file)
    print('parameter has saved.')
    fout.close()

