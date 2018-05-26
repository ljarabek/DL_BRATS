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
        ema = tf.train.ExponentialMovingAverage(decay=0.9) #prej je blo 0.5

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
    features = int(2*features_input)
    output_halved = conv3D(input, features, 2,3, name = name + "_conv/2")
    output1 = batch_norm(output_halved,features,phase_train=phase_train)

    output1 = prelu(output1)
    output1 = batch_norm(output1, features, phase_train=phase_train)
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

def interpolation(input, output_shape, name=None):
    """
    :param input: data
    :param output_shape: [batch x*2 y*2 z*2 channels]
    :param name: name
    :return: interpolated matrix (double the size)
    """
    A= tf.constant(value=[[[0.125, 0.25, 0.125],     [0.25, 0.5, 0.25],     [0.125, 0.25, 0.125]],
                          [[0.25 , 0.5 , 0.25],      [0.50, 1.00, 0.50],    [0.25, 0.5, 0.25]],
                          [[0.125, 0.25, 0.125],     [0.25, 0.5, 0.25],     [0.125, 0.25, 0.125]]],
                   name = "interp_kernel_"+name)
    A = tf.expand_dims(A, axis = 3)
    #print(tf.shape(A))
    A = tf.expand_dims(A, axis=4)

    A_ = tf.constant(value=[[[0.125, 0.25, 0.125], [0.25, 0.5, 0.25], [0.125, 0.25, 0.125]],
                           [[0.25, 0.5, 0.25], [0.50, 1.00, 0.50], [0.25, 0.5, 0.25]],
                           [[0.125, 0.25, 0.125], [0.25, 0.5, 0.25], [0.125, 0.25, 0.125]]],
                    name="interp_ke_" + name)
    A_ = tf.expand_dims(A_, axis=3)
    # print(tf.shape(A))
    A_ = tf.expand_dims(A_, axis=4)

    #print(tf.shape(A))
    print(input.get_shape())
    b, x, y, z, ch= output_shape
    for i in range(int(ch/2)-1):
        A = tf.concat([A, A], axis = 3)
        A = tf.concat([A, A], axis=4)
    print("A + {}".format(A.get_shape()))
    return tf.nn.conv3d_transpose(input, filter=A,output_shape = output_shape,strides=[1,2,2,2,1], padding= "SAME", name =  "interp_"+name)
def prelu(input, alphax = 0.2):
    alpha = tf.Variable(alphax, True)
    return tf.nn.leaky_relu(input, alpha=alpha)

def expandingBlock(input, skip_input,phase_train, features_input, name = "expand_block", concat_inputs = False,
                   concatenation = False):

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

"""def interpolation(data):
    sh = data.shape
    n = int(sh[4])
    width, height, depth = int(sh[1]), int(sh[2]), int(sh[3])
    batch_size = 1
    #data = tf.transpose(data, [0, 4, 1, 2, 3])  # [batch_size,n,width,height,depth]
    newshape = int(1 * n * width * height * depth)
    flatten_it_all = tf.reshape(data, shape = [newshape, 1])  # flatten it
    print(flatten_it_all.shape)  #(23592960, 1)
    expanded_it = flatten_it_all * tf.ones([1, 8])
    print(expanded_it.shape)  # (23592960, 8)
    prepare_for_transpose = tf.reshape(expanded_it, [batch_size * n, width, height, depth, 2, 2, 2])
    print(prepare_for_transpose.shape)
    transpose_to_align_neighbors = tf.transpose(prepare_for_transpose, [0, 1, 6, 2, 5, 3, 4])
    print(transpose_to_align_neighbors.shape)
    expand_it_all = tf.reshape(transpose_to_align_neighbors, [batch_size, n, width * 2, height * 2, depth * 2])
    expand_it_all = tf.transpose(expand_it_all, [0,2,3,4,1])
    #### - removing this section because the requirements changed
    # do a conv layer here to 'blend' neighbor values like:
    averager = tf.ones([2,2,2,8,8]) * 1. / 8.
    expand_it_all = tf.nn.conv3d( expand_it_all , strides = [1,2,2,2,1], filter = averager , padding="SAME")
    # for n = 1.  for n = 3, I'll leave it to you.

    # then finally reorder and you are done
    #return expand_it_all # tf.transpose(expand_it_all, [0, 2, 3, 4, 1])
    return prepare_for_transpose"""




def interpolationxxx(data, scale = 2):
    # yourData shape : [5,50,50,10,256] to [5,100,100,10,256]
    # [1, x,y,z, channels]
    # First reorder your dimensions to place them where tf.image.resize_images needs them

    transposed = tf.transpose(data, [0, 3, 1, 2, 4])
    sh = transposed.shape
    # it is now [5,10,50,50,256]
    # but we need it to be 4 dimensions, not 5
    reshaped = tf.reshape(transposed, [sh[1]*sh[0], sh[2], sh[3], sh[4]]) # [5*10,50,50,256]

    # and finally we use tf.image.resize_images
    #new_size = tf.constant( [sh[2], sh[3]] * scale)
    sh2 = sh[2]*2
    sh3 = sh[3]*2
    print(sh2, sh3)
    resized = tf.image.resize_images(reshaped, (sh2, sh3))

    # your data is now [5*10,100,100,256]
    undo_reshape = tf.reshape(resized, [sh[0], sh[1]])

    # it is now [5,10,100,100,256] so lastly we need to reorder it
    undo_transpose = tf.transpose(undo_reshape, [0, 2, 3, 1, 4])
    return undo_transpose



