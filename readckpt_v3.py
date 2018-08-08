from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import numpy as np
import re
import ctypes

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# 读取权重，并且输出为各个权重和偏置的txt

FLAGS = None
save_file='./ckptlenet/last.ckpt'

reader=tf.train.NewCheckpointReader(save_file)
variables=reader.get_variable_to_shape_map()

mask = 0xffff
for element in variables:
    if 'Adam' not in element and 'beta' not in element:
        #该element就是相应的权重和偏置
        getarray = reader.get_tensor(element)
        getarray = getarray * 1024
        getarray = getarray.astype(np.int16)
        shape = getarray.shape
        print(element)
        print(shape)
        ele_text = re.sub('/','_',element)
        print(ele_text)
        f =  open(ele_text + ".txt",'w')
        f_st =  open(ele_text + "_st.txt",'w')
        #分情况考虑
        if '/L' in element and 'weight' in element:
            for box in range(shape[3]):
                f_st.write("NUM %02d BOX:\n" % box)
                for channel in range(shape[2]):
                    f_st.write("Channel %02d:\n" % channel)
                    for line in range(shape[1]):
                        f_st.write("%02d:" % line)
                        for pixel in range(shape[0]):
                            num = getarray[line,pixel,channel,box]
                            num = "%04x" % (ctypes.c_uint32(num).value & mask)
                            f.write(num + "\n")
                            f_st.write(" " + num)
                        f_st.write("\n")
        if '/F1' in element and 'weight' in element:
            #对于F1，是先所有输出，先所有盒子的左上角，再纵着排布，再横向排布
            for onode in range(shape[1]):
                f_st.write("OUTPUT %03d:\n" % onode)
                for channel in range(16):
                    #f_st.write("Channel %02d:\n" % channel)
                    for line in range(5):
                        #f_st.write("%02d:" % line)
                        for pixel in range(5):
                            num = getarray[line*80+pixel*16+channel,onode]
                            num = "%04x" % (ctypes.c_uint32(num).value & mask)
                            f.write(num + "\n")
                            #f_st.write(" " + num)
                            f_st.write(num + "\n")
                        #f_st.write("\n")
        elif '/F2' in element and 'weight' in element:
            for onode in range(shape[1]):
                for inode in range(shape[0]):
                    num = getarray[inode,onode]
                    num = "%04x" % (ctypes.c_uint32(num).value & mask)
                    f.write(num + "\n")
        elif '/F3' in element and 'weight' in element:
            for onode in range(shape[1]):
                for inode in range(shape[0]):
                    num = getarray[inode,onode]
                    num = "%04x" % (ctypes.c_uint32(num).value & mask)
                    f.write(num + "\n")
        elif 'bias' in element:
            for pixel in range(shape[0]):
                num = getarray[pixel]
                num = "%04x" % (ctypes.c_uint32(num).value & mask)
                f.write(num + "\n")

        f.close()
        f_st.close()
