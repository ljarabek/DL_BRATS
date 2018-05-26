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

input = tf.placeholder(dtype = tf.float32, shape = _input.shape)
phase_train = tf.placeholder(dtype=tf.bool)
#Ar = tf.reshape(A, shape=(1, 128, 160, 144, 4))

answer = tf.placeholder(dtype = tf.float32)

Ar = tf.transpose(input, perm = [0,2,3,4,1]) #(1, 4, 128, 160, 144)  -->  1 index in inpuza hočem d aje na 2-jki(1, 128, 160, 144, 4)



Ar = conv3D(Ar, 8, 1, 3, name = "preUnet")
Ar = batch_norm(Ar, 8, phase_train)
Ar = prelu(Ar, 0.2)

"""Ar = Ar[:,:,:,:,0]
print("arshape: {}".format(Ar.shape))
Ar = tf.reshape(Ar, shape = (128*160*144,1))#shape= (1, 128*2, 160*2, 144*2))
Ar = Ar * tf.ones([1,8])
Ar = tf.reshape(Ar, shape= (1, 128*2 ,160*2,  2*144,1))"""

Ar = interpolation(Ar,output_shape=[1, 256, 320, 288, 8], name="fuck_it")


"""U-NET"""
"""contr1 = contractingBlock(Ar,phase_train,8,name="contr_1")
contr2 = contractingBlock(contr1, phase_train, 16, name="contr_2")
contr3 = contractingBlock(contr2, phase_train, 32, name="contr_3")

exp1 = expandingBlock(contr3, contr2, phase_train, 64, name="exp_1")
exp2 = expandingBlock(exp1, contr1, phase_train, 32, name="exp_2")
exp3 = expandingBlock(exp2, Ar, phase_train, 16, name="exp_3")"""

"""TISTO ZRAVEN:"""

#interp1 = interpolation(Ar)


#Ar = interpolation(Ar)






output =  tf.transpose(Ar, [0,4,1,2,3])
#output = tf.transpose(Ar, [0, 1, 6, 2, 5, 3, 4])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    otpt = sess.run(output, feed_dict={input:_input, phase_train:True})
    print(otpt.shape)
    #print(_input.shape)
    #sha = sess.run(output, feed_dict={A:_input, B: _answer[0:4]})
    display_numpy(_input[0, 2, :, :, :])
    display_numpy(otpt[0, 0, :, :, :])
    #display_numpy(otpt[0,:,0,:,0,:,0])
    plt.show()

#plt.imshow(sha[3][64])
#plt.show()