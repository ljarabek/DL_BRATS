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

from modelTesting import train_model


"""III IMPORTANT III"""




DCT = dictionary.get()
print(len(DCT))
seed = 42
l2_regularization = 0.0


"""READ CONFIG"""
cp = ConfigParser()
cp.read("config.ini")
cfg = cp['DEFAULT']
#val_size = int(cfg['val_size'])
for d in cfg:
    exec("%s = %s"%(d, cfg[d]))     # loads all variables from config.ini DEFAULT

_input, _answer = getBatchTraining()

print(_input.shape)  # (1, 4, 128, 160, 144)

input = tf.placeholder(dtype=tf.float32, shape=_input.shape, name="input")
answer = tf.placeholder(dtype=tf.float32, shape=_answer.shape, name="answer")
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

loss = jaccard_coef_logloss(output, answer, name="loss")
tf.summary.scalar("loss", loss)
LR = tf.Variable(initial_value=0.001, dtype=tf.float32, trainable=False, name="learning_rate")

train = tf.train.AdamOptimizer(learning_rate=LR, name="train").minimize(loss)




for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
tf.summary.scalar("learning_rate", LR)
tf.summary.histogram("contr1", contr1)
tf.summary.histogram("contr2", contr2)
tf.summary.histogram("contr3", contr3)
tf.summary.histogram("exp1", exp1)
tf.summary.histogram("exp2", exp2)
tf.summary.histogram("exp3", exp3)

merged = tf.summary.merge_all()           # black magic

sess = tf.Session()

# grid search noises
loader = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(checkpoint_dir='C:\modelLogs\main_0.2_1533311066.683526\ckpt_0.2_1533311066.683526')
load(loader, sess, ckpt.model_checkpoint_path)

_input, ID = getBatchTest(True)
otpt, summary = sess.run([output, merged], feed_dict={input: _input, phase_train: True, answer: _answer})
#otpt = channelsToOutput(otpt)
otpt = otpt[0]
saveSegmentation(_input[0, 2], otpt, 10,"c:/seg/")


"""
yey

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())                                 # initialize variables
    train_writer = tf.summary.FileWriter('C:/train/', sess.graph)               # write graph to tensorboard
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)         # loads existing checkpoint

    top_loss_step = 0   # batch step where best loss was achieved

    if ckpt and ckpt.model_checkpoint_path:     # loads ckpt if it exists and
        loader = tf.train.Saver()               # obtains latest learning rate of the model
        load(loader, sess, ckpt.model_checkpoint_path)
        learning_rate = sess.run([LR], feed_dict={input: _input, phase_train: True, answer: _answer})
        learning_rate = learning_rate[0]
    else:
        learning_rate = lr_0
    top_loss = 1e5      # placeholder for lowest loss
    for i in range(batches):
        _input, _answer = getBatchTraining()

        if _train:
            otpt, loss_, summary, _ = sess.run([output, loss, merged, train],
                                               feed_dict={input: _input, phase_train: True, answer: _answer,
                                                          LR: learning_rate})

            if i%50==0:
                for i in range(val_size):
                    _input, _answer = getBatchVal()
                    otpt, summary = sess.run([output, merged],
                                             feed_dict={input: _input, phase_train: True, answer: _answer})

            otpt = otpt[0]
            train_writer.add_summary(summary, i)
            if (i+1) % lr_rate == 2490:
                learning_rate = learning_rate * lr_multiplier
            if i > 500 and loss_ < top_loss and i % 50 == 0:
                print("saving top results, loss: {}".format(loss_))
                top_loss = loss_
                top_loss_step = i
                save(saver, sess, checkpoint_dir, i)


            #TIMEOUT:
            if i>top_loss_step+4000:
                print("No progress in last 4k steps, stopping training")
                break
            print(loss_, learning_rate)
        else:
            _input, ID = getBatchTest(True)
            otpt, summary = sess.run([output, merged],
                                     feed_dict={input: _input, phase_train: True, answer: _answer})
            otpt = otpt[0]

            saveSegmentation(_input[0, 2], otpt, i,"c:/new_segmentations/")

    train_writer.close()
    print(otpt.shape)
    print(loss_)
"""