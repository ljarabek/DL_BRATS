import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
from preprocessing import *
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


def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5) #prej je blo 0.5

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def conv3D(input, features, stride = 1,kernel = 3, name=None):
    #weight = tf.get_variable(name="conv3d_"+name, shape=[])
    return tf.layers.conv3d(inputs = input, filters = features, kernel_size = [kernel,kernel,kernel], strides=[stride, stride, stride], padding = "same",
                     activation=None, use_bias = False, #data_format="channels_first"
                     kernel_initializer = xavier_initializer(uniform = False, seed = seed),
                     kernel_regularizer = l2_regularizer(l2_regularization), trainable=True, name= "conv3d_" + name)


def contractingBlock(input, phase_train, features_input, name = "_contrblock"):
    with tf.variable_scope(name):
        features = int(2*features_input)
        output_halved = conv3D(input, features, 2,3, name = name + "_conv/2")
        output1 = batch_norm(output_halved,features,phase_train=phase_train)

        output1 = prelu(output1)
        #output1 = batch_norm(output1, features, phase_train=phase_train)
        output1 = conv3D(output1, features=features)
        output_block = output1+output_halved
        output_block = batch_norm(output_block,features,phase_train=phase_train)

        output_block = prelu(output_block)
    return output_block


def deconv3D(input, features, stride = 2,kernel = 3, name=None):
    return tf.layers.conv3d_transpose(inputs=input, filters=features, kernel_size=[kernel,kernel,kernel],
                                      strides=[stride,stride,stride], padding="same",
                                      kernel_initializer=xavier_initializer(uniform=False, seed=seed),
                                      use_bias = False,
                                      kernel_regularizer=l2_regularizer(l2_regularization), trainable=True,
                                      name="deconv3d_" + name)

def interpolation(input,  name="interp_2x", no_filters = 4): #output_shape,
    """
    :param input: data
    :param output_shape: [batch x*2 y*2 z*2 channels]
    :param name: name
    :return: interpolated matrix (double the size)
    """
    with tf.variable_scope(name):
        A= tf.constant(value=[[[0.125, 0.25, 0.125],     [0.25, 0.5, 0.25],     [0.125, 0.25, 0.125]],
                              [[0.25 , 0.5 , 0.25],      [0.50, 1.00, 0.50],    [0.25, 0.5, 0.25]],
                              [[0.125, 0.25, 0.125],     [0.25, 0.5, 0.25],     [0.125, 0.25, 0.125]]],
                       name = "interp_kernel_"+name)
        A = tf.expand_dims(A, axis = 3)
        #print(tf.shape(A))
        A = tf.expand_dims(A, axis=4)
        fil= tf.concat([A, A], axis = 3,name="conc5")
        buffer = tf.concat([A,A], axis=3,name="conc4")
        #B = tf.concat([B, A], axis=4)
        for i in range(no_filters-2):
            fil = tf.concat([fil,A], axis=3, name="conc")
            buffer = tf.concat([buffer, A], axis=3,name="conc1")
            #B = tf.concat([B, A], axis=4)
        for i in range(no_filters-1):
            fil = tf.concat([fil, buffer], axis = 4,name="conc2")
        #A = tf.stack([A, A, A, A, A], axis=3)
        #A = tf.stack([A, A, A, A, A], axis=4)
        #print(tf.shape(A))

        b_, x_, y_, z_, ch_ = list(input.get_shape())
        x_ *=2
        y_ *=2
        z_ *=2
        #print("channels for interp: "+ str(x_))
        #b, x, y, z, ch= output_shape

        #for i in range(int(int(ch_)/2)):
        #B = tf.concat([A, A], axis = 3)
        #B = tf.concat([A, A], axis=4)
        #print("A + {}".format(A.get_shape()))
        output_shape = tf.constant([b_,x_,y_,z_,ch_], dtype= tf.int32)
        #output_shape = tf.convert_to_tensor(output_shape)
    return tf.nn.conv3d_transpose(input, filter=fil,output_shape = output_shape,strides=[1,2,2,2,1], padding= "SAME", name =  "interp_"+name)
def prelu(input, alphax = 0.2):
    with tf.variable_scope("prelu"):
        alpha = tf.Variable(alphax, True, name="alpha")
        return tf.nn.leaky_relu(input, alpha=alpha)

def expandingBlock(input, skip_input,phase_train, features_input, name = "expand_block", concat_inputs = False,
                   concatenation = False):
    with tf.variable_scope(name):
        if concat_inputs:
            features = int(features_input/4)
        else:
            features = int(features_input/2)
        output = conv3D(input,features=features, stride=1, kernel=1,name=name + "_1x1x1conv")
        output = batch_norm(output, features,phase_train=phase_train)

        output = prelu(output)
        output= deconv3D(output, features,stride=2, kernel=3, name=name + "_deconv")
        output = batch_norm(output, features, phase_train=phase_train)
        output = prelu(output)
        if concatenation:
            output = tf.concat([output, skip_input], axis=4)
        else:
            output = output + skip_input
        output = conv3D(output, features_input,stride=1,kernel=3,name=name + "_conv3D")
        output = batch_norm(output, features_input, phase_train=phase_train)
        output = prelu(output)

    return output





