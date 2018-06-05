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
from loss_function import *
from layers import *

DCT = dictionary.get()
print(len(DCT))
seed = 42
l2_regularization = 0.0

_input, _answer = getBatchTraining()
#display_numpy(_answer[0, 0, :, :, :])
#_input = np.expand_dims(_input, 0)
#_answer = np.expand_dims(_answer, 0)
print(_input.shape)  #  (1, 4, 128, 160, 144)  -->  (1, 128, 160, 144, 4)

input = tf.placeholder(dtype = tf.float32, shape = _input.shape)
answer = tf.placeholder(dtype=tf.float32, shape=_answer.shape)
phase_train = tf.placeholder(dtype=tf.bool)
#Ar = tf.reshape(A, shape=(1, 128, 160, 144, 4))


Ar = tf.transpose(input, perm = [0,2,3,4,1]) #(1, 4, 128, 160, 144)  -->  1 index in inpuza hočem d aje na 2-jki(1, 128, 160, 144, 4)



Ar = conv3D(Ar, 8, 1, 3, name = "preUnet")
Ar = batch_norm(Ar, 8, phase_train)
Ar = prelu(Ar, 0.2)  # 2*shape = [1, 256, 320, 288, 8]

"""Ar = Ar[:,:,:,:,0]
print("arshape: {}".format(Ar.shape))
Ar = tf.reshape(Ar, shape = (128*160*144,1))#shape= (1, 128*2, 160*2, 144*2))
Ar = Ar * tf.ones([1,8])
Ar = tf.reshape(Ar, shape= (1, 128*2 ,160*2,  2*144,1))"""

#Ar = interpolation(Ar,output_shape=[1, 256, 320, 288, 8], name="fuck_it")


"""U-NET"""
contr1 = contractingBlock(Ar,phase_train,8,name="contr_1")
contr2 = contractingBlock(contr1, phase_train, 16, name="contr_2")
contr3 = contractingBlock(contr2, phase_train, 32, name="contr_3")

exp1 = expandingBlock(contr3, contr2, phase_train, 64, name="exp_1")
exp2 = expandingBlock(exp1, contr1, phase_train, 32, name="exp_2")
exp3 = expandingBlock(exp2, Ar, phase_train, 16, name="exp_3")

"""TISTO ZRAVEN:"""
out1 = conv3D(exp1, 5, 1, 1, name="1x1x1_conv_1")
out1 = interpolation(out1, no_filters=5)
out2 = conv3D(exp2, 5, 1, 1, name="1x1x1_conv_2")
out2 = out2+out1
out2 = interpolation(out2, no_filters=5)
out3 = conv3D(exp3, 5, 1, 1, name="1x1x1_conv_3")
out3 = out3 + out2
out3 = tf.nn.softmax(out3, dim=-1, name="softmax" )  # DEJMO TO PRESKOČIT, ČE NIMAMO ZA VSAK PIXEL LABEL!!
#out3 = tf.nn.sigmoid(out3)
#Ar = interpolation(Ar)
#Ar = interpolation(Ar)
#interp1 = interpolation(Ar)


#Ar = interpolation(Ar)






output =  tf.transpose(out3, [0,4,1,2,3])
print(output.get_shape())
print(answer.get_shape())
#loss = jaccard_coef_logloss(output, answer)
loss = jaccard_coef_logloss(output, answer)
LR = tf.placeholder(tf.float32)
train = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)
#output = tf.transpose(Ar, [0, 1, 6, 2, 5, 3, 4])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter( 'C:/train/', sess.graph)
    learning_rate = 0.006
    top_loss = 1e5
    for i in range(50000):
        _input, _answer = getBatchTraining()
        otpt, loss_, _ = sess.run([output, loss, train], feed_dict={input:_input, phase_train:True, answer: _answer, LR:learning_rate})
        if i%2500==0:
            learning_rate = learning_rate/2
        if i%100==0 and loss_ < top_loss:
            print("saving top results, loss: {}".format(loss_))
            top_loss=loss_

            save_numpy(_input[0, 0, :, :, :], i, filename="in_flair.png")
            save_numpy(_input[0, 1, :, :, :], i, filename="in_t1.png")
            save_numpy(_input[0, 2, :, :, :], i, filename="in_t1c.png")
            save_numpy(_input[0, 3, :, :, :], i, filename="in_t2.png")
            save_numpy(otpt[0, 0, :, :, :], i, filename = "out_0.png")
            save_numpy(otpt[0, 1, :, :, :], i, filename="out_1.png")
            save_numpy(otpt[0, 2, :, :, :], i, filename = "out_2.png")
            save_numpy(otpt[0, 3, :, :, :], i, filename="out_3.png")
            save_numpy(otpt[0, 4, :, :, :], i, filename="out_4.png")
            save_numpy(_answer[0, 0, :, :, :], i, filename = "answer_0.png")
            save_numpy(_answer[0, 1, :, :, :], i, filename="answer_1.png")
            save_numpy(_answer[0, 2, :, :, :], i, filename="answer_2.png")
            save_numpy(_answer[0, 3, :, :, :], i, filename = "answer_3.png")
            save_numpy(_answer[0, 4, :, :, :], i, filename="answer_4.png")  # tle je bil anwer3.png!!! napaka!
            np.save("C:/activations/{}input.npy".format(i), _input)
            np.save("C:/activations/{}output.npy".format(i), otpt)
            np.save("C:/activations/{}output.npy".format(i), _answer)
            #save_numpy(_answer[0, 4, :, :, :], i, filename="answer_4.png")
            #save_numpy(_answer[0, 5, :, :, :], i, filename="answer_3.png")
        print(loss_, learning_rate)
    train_writer.close()
    print(otpt.shape)
    print(loss_)
    #print(_input.shape)
    #sha = sess.run(output, feed_dict={A:_input, B: _answer[0:4]})
    """display_numpy(_input[0, 2, :, :, :])
    display_numpy(otpt[0, 1, :, :, :])
    display_numpy(_answer[0, 1, :, :, :])"""
    #display_numpy(otpt[0,:,0,:,0,:,0])


#plt.imshow(sha[3][64])
#plt.show()