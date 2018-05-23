import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
from testing import *
import dictionary
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import random as rand
from tqdm import tqdm
from patient import Patient
import os
from loss_function import jaccard_loss
seed = 42
l2_regularization = 0.0

_input, _answer = getBatchTraining()
_input = np.expand_dims(_input, 0)
print(_input.shape)  #  (1, 4, 128, 160, 144)  -->  (1, 128, 160, 144, 4)

A = tf.placeholder(dtype = tf.float32, shape = _input.shape)
#Ar = tf.reshape(A, shape=(1, 128, 160, 144, 4))

Ar = tf.transpose(A, perm = [0,2,3,4,1]) #(1, 4, 128, 160, 144)  -->  1 index in inpuza hočem d aje na 2-jki(1, 128, 160, 144, 4)
print(Ar.shape)
B = tf.placeholder(dtype = tf.float32)

def conv3D(input, name, stride = 1):
    #weight = tf.get_variable(name="conv3d_"+name, shape=[])
    return tf.layers.conv3d(inputs = input, filters = 8, kernel_size = [3,3,3], strides=[stride, stride, stride], padding = "same",
                     activation=tf.nn.leaky_relu, use_bias = False, #data_format="channels_first"
                     kernel_initializer = xavier_initializer(uniform = False, seed = seed),
                     kernel_regularizer = l2_regularizer(l2_regularization), trainable=True, name= "conv3d_" + name)



output1 = tf.transpose(conv3D(Ar, "ime"), [0,4,1,2,3])  # (1, 128, 160, 144, 4)  -->  (1, 4, 128, 160, 144)



output2 = conv3D(output1, "ime2", stride = 2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    otpt = sess.run(output1, feed_dict={A:_input})
    print(otpt.shape)
    #print(_input.shape)
    #sha = sess.run(output, feed_dict={A:_input, B: _answer[0:4]})
    plt.imshow(_input[0, 2, :, :, 64])
    plt.show()
    plt.imshow(otpt[0, 2, :, :, 64])
    plt.show()

#plt.imshow(sha[3][64])
#plt.show()