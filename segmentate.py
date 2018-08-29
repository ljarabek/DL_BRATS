import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
from trainingUtils import *
import dictionary
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import random as rand
from tqdm import tqdm
import os
from loss_function import *
from layers import *
from preprocessing import *
from postprocessing import *
from tqdm import tqdm
from presentation import seg_viewer, multi_slice_viewer
from modelTesting import train_model

DCT = dictionary.get()



"""READ CONFIG"""
cp = ConfigParser()
cp.read("config.ini")
cfg = cp['DEFAULT']
#val_size = int(cfg['val_size'])
for d in cfg:
    exec("%s = %s"%(d, cfg[d]))     # loads all variables from config.ini DEFAULT



def segmentate(flair, t1,t1c,t2):
    """

    :param flair:  path to .mha flair
    :param t1:  path to .mha t1
    :param t1c:  path to .mha t1c
    :param t2:  path to .mha t2
    :return:  segmentation according to BRATS2015 rules
    """
    _input= getInput(flair,t1,t1c,t2)

    print(_input.shape)  # (1, 4, 128, 160, 144)

    input = tf.placeholder(dtype=tf.float32, shape=_input.shape, name="input")
    phase_train = tf.placeholder(dtype=tf.bool, name="phase_train")



    Ar = tf.transpose(input, perm=[0, 2, 3, 4, 1])  # (1, 4, 128, 160, 144)  -->  (1, 128, 160, 144, 4)

    Ar = conv3D(Ar, 8, 1, 3, name="preUnet")
    Ar = batch_norm(Ar, 8, phase_train)
    Ar = prelu(Ar, 0.2)  # 2*shape = [1, 256, 320, 288, 8]


    """U-NET"""
    contr1 = contractingBlock(Ar, phase_train, 8, name="contr_1")
    contr2 = contractingBlock(contr1, phase_train, 16, name="contr_2")
    contr3 = contractingBlock(contr2, phase_train, 32, name="contr_3")

    exp1 = expandingBlock(contr3, contr2, phase_train, 64, name="exp_1")
    exp2 = expandingBlock(exp1, contr1, phase_train, 32, name="exp_2")
    exp3 = expandingBlock(exp2, Ar, phase_train, 16, name="exp_3")

    """interpolating path:"""
    out1 = conv3D(exp1, 5, 1, 1, name="1x1x1_conv_1")  # reduce channels to 5
    out1 = interpolation(out1, no_filters=5)
    out2 = conv3D(exp2, 5, 1, 1, name="1x1x1_conv_2")
    out2 = out2 + out1
    out2 = interpolation(out2, no_filters=5)
    out3 = conv3D(exp3, 5, 1, 1, name="1x1x1_conv_3")
    out3 = out3 + out2
    out3 = tf.nn.softmax(out3, dim=-1, name="softmax")

    output = tf.transpose(out3, [0, 4, 1, 2, 3], name="output")

    LR = tf.Variable(initial_value=0.001, dtype=tf.float32, trainable=False, name="learning_rate")

    sess = tf.Session()

    # grid search noises
    loader = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir='C:\modelLogs\main_0.2_1533311066.683526\ckpt_0.2_1533311066.683526')
    load(loader, sess, ckpt.model_checkpoint_path)


    #_input, ID = getBatchTest(True)
    otpt = sess.run([output], feed_dict={input: _input, phase_train: True})
    otpt= otpt[0][0]
    #otpt = channelsToOutput(otpt)
    saveSegmentation(_input[0, 2], otpt, 10,"c:/seg/")

    otpt = channelsToOutput(otpt)
    saveSeg(_input[0,2], otpt,11,"c:/seg/")

    print(np.shape(otpt))

    return otpt

#multi_slice_viewer(np.array(resizeOutput(segmentate(DCT[15][0],DCT[15][1],DCT[15][2],DCT[15][3]))))
answer = segmentate(DCT[11][0],DCT[11][1],DCT[11][2],DCT[11][3])
#print(np.array(getInput(DCT[11][0],DCT[11][1],DCT[11][2],DCT[11][3])[0,2]).shape)
seg_viewer(np.array(getInput(DCT[11][0],DCT[11][1],DCT[11][2],DCT[11][3])[0,2]),np.array(answer))