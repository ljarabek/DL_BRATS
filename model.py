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
from layers import *

seed = 42
l2_regularization = 0.0

_input, _answer = getBatchTraining()
_input = np.expand_dims(_input, 0)
print(_input.shape)  #  (1, 4, 128, 160, 144)  -->  (1, 128, 160, 144, 4)

A = tf.placeholder(dtype = tf.float32, shape = _input.shape)
phase_train = tf.placeholder(dtype=tf.bool)
#Ar = tf.reshape(A, shape=(1, 128, 160, 144, 4))

Ar = tf.transpose(A, perm = [0,2,3,4,1]) #(1, 4, 128, 160, 144)  -->  1 index in inpuza hočem d aje na 2-jki(1, 128, 160, 144, 4)
print(Ar.shape)
B = tf.placeholder(dtype = tf.float32)








output = contractingBlock(Ar,phase_train,8,name="test")
output = contractingBlock(output,phase_train,16,name= "haha")
deconv = deconv3D(output, 8, 2, name="mmhh")
deconv = deconv3D(deconv, 4, 2, name="mmdhh")
output = deconv + Ar
output =  tf.transpose(output, [0,4,1,2,3])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    otpt = sess.run(output, feed_dict={A:_input, phase_train:True})
    print(otpt.shape)
    #print(_input.shape)
    #sha = sess.run(output, feed_dict={A:_input, B: _answer[0:4]})
    display_numpy(_input[0, 2, :, :, :])

    display_numpy(otpt[0, 2, :, :, :])
    plt.show()

#plt.imshow(sha[3][64])
#plt.show()