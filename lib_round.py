import numpy as np
import tensorflow as tf
import ctypes

def conv2d(x, W):
    b_x,h_x,w_x,c_x = x.shape
    h_w,w_w,c_w,n_w = W.shape

    #b_x = int(b_x)
    #h_x = int(h_x)
    #w_x = int(w_x)
    #c_x = int(c_x)
    #h_w = int(h_w)
    #w_w = int(w_w)
    #c_w = int(c_w)
    #n_w = int(n_w)

    h_y = h_x-h_w+1
    w_y = w_x-w_w+1

    y = np.zeros([b_x,h_y,w_y,n_w],dtype=np.int16)
    for b in range(b_x):
        print("img %d" % b)
        for h in range(h_y):
            for w in range(w_y):
                for n in range(n_w):
                    #计算输出的第[b,h,w,n]个点的值
                    mula = 0
                    for hn in range(h_w):
                        for wn in range(w_w):
                            for cn in range(c_w):
                                mulx = int(x[b,h+hn,w+wn,cn])
                                mulw = int(W[hn,wn,cn,n])
                                mulresult = mulx * mulw
                                mulresult = mulresult//1024
                                mula = mula + mulresult
                    y[b,h,w,n] = mula
    return y
    #return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
def conv2d_same(x, W):
    b_x,h_x,w_x,c_x = x.shape
    h_w,w_w,c_w,n_w = W.shape

    #b_x = int(b_x)
    #h_x = int(h_x)
    #w_x = int(w_x)
    #c_x = int(c_x)
    #h_w = int(h_w)
    #w_w = int(w_w)
    #c_w = int(c_w)
    #n_w = int(n_w)

    h_shift = h_w//2
    w_shift = w_w//2

    y = np.zeros([b_x,h_x,w_x,n_w],dtype=np.int16)
    for b in range(b_x):
        print("img %d" % b)
        for h in range(h_x):
            for w in range(w_x):
                for n in range(n_w):
                    #计算输出的第[b,h,w,n]个点的值
                    mula = 0
                    for hn in range(h_w):
                        for wn in range(w_w):
                            for cn in range(c_w):
                                #如果超出边界则置0
                                if h-h_shift+hn < 0 or h-h_shift+hn >= h_x or w-w_shift+wn < 0 or w-w_shift+wn >= w_x:
                                    mulx = 0
                                else:
                                    mulx = int(x[b,h-h_shift+hn,w-w_shift+wn,cn])
                                mulw = int(W[hn,wn,cn,n])
                                mulresult = mulx * mulw
                                mulresult = mulresult//1024
                                mula = mula + mulresult
                    y[b,h,w,n] = mula
    return y
    #return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def matmul(x,W):
    b_x,n_x = x.shape
    nx_w,ny_w = W.shape

    #b_x = int(b_x)
    #n_x = int(n_x)
    #nx_w = int(nx_w)
    #ny_w = int(ny_w)
    y = np.zeros([b_x,ny_w],dtype=np.int16)
    for b in range(b_x):
        print("img %d" % b)
        for ny in range(ny_w):
            #计算输出的第[b,ny]个点的值
            mula = 0
            for nx in range(nx_w):
                mulx = int(x[b,nx])
                mulw = int(W[nx,ny])
                mulresult = mulx * mulw
                mulresult = mulresult//1024
                mula = mula + mulresult
            y[b,ny] = mula
    return y

def conv_add(x, B):
    b_x,h_x,w_x,c_x = x.shape

    for bn in range(b_x):
        for h in range(h_x):
            for w in range(w_x):
                for c in range(c_x):
                    x[bn,h,w,c] = x[bn,h,w,c] + B[c]
    return x

def max_pool_2x2(x):
    b_x,h_x,w_x,c_x = x.shape

    h_x_2 = h_x//2
    w_x_2 = w_x//2

    y = np.zeros([b_x,h_x_2,w_x_2,c_x],dtype=np.int16)
    for b in range(b_x):
        for h in range(h_x_2):
            for w in range(w_x_2):
                for c in range(c_x):
                    maxn = 0
                    for hn in range(2):
                        for wn in range(2):
                            maxn = max(maxn,x[b,2*h+hn,2*w+wn,c])
                    y[b,h,w,c] = maxn
    return y

def relu(x):
    shape = x.shape
    x_tmp = x.reshape(-1)
    x_tmp = np.array([max(x,0) for x in x_tmp])
    return x_tmp.reshape(shape)

def change_int(array):
    x1024 = array * 1024
    return x1024.astype(np.int16)

