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

cp = ConfigParser()
cp.read("config.ini")
cfg = cp['CROP']
# val_size = int(cfg[''])
for d in cfg:
    # print(cfg[d])
    exec("%s = %s" % (d, cfg[d]))


def train_model(sess):
    #global merged
    _input, _answer = getBatchTraining()
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())  # initialize variables
    train_writer = tf.summary.FileWriter('C:/train/', sess.graph)  # write graph to tensorboard
    val_writer = tf.summary.FileWriter('C:/val/', sess.graph)  # write graph to tensorboard
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)  # loads existing checkpoint
    val_loss = 0.00
    val_loss_top = 1e5
    losses = []
    top_loss_step = 0  # batch step where best loss was achieved

    if ckpt and ckpt.model_checkpoint_path:  # loads ckpt if it exists and
        loader = tf.train.Saver()  # obtains latest learning rate of the model
        load(loader, sess, ckpt.model_checkpoint_path)
        learning_rate = sess.run(["learning_rate:0"], feed_dict={"input:0": _input, "phase_train:0": True, "answer:0": _answer})
        learning_rate = learning_rate[0]
    else:
        learning_rate = lr_0
    top_loss = 1e5  # placeholder for lowest loss
    for i in range(batches):
        _input, _answer = getBatchTraining()


        otpt, loss_, summary, _ = sess.run(["output:0", "loss:0", merged, "train"],
                                           feed_dict={"input:0": _input, "phase_train:0": True, "answer:0": _answer,
                                                      "learning_rate:0": learning_rate})

        if i % save_losses_every == 0:
            for dfdskjfmsdfm in range(val_size):
                _input, _answer = getBatchVal()
                otpt, summary_val, v_loss = sess.run(["output:0", merged, "loss:0"],
                                         feed_dict={"input:0": _input, "phase_train:0": True, "answer:0": _answer})
                val_loss += v_loss/float(val_size)
            losses.append([loss_, val_loss])

            val_writer.add_summary(summary_val)
            print(losses)
            np.save("train_val_loss.npy", arr=losses)
            print("validation loss: %s , training loss: %s" % (val_loss, loss_))



            if i > top_loss_b and loss_ < top_loss and val_loss<val_loss_top:

                top_loss = loss_
                val_loss_top = val_loss
                print("saving top results, train loss: %s validation loss: %s" % (top_loss, val_loss_top))
                top_loss_step = i
                save(saver, sess, checkpoint_dir, i)        # save model at best loss

            val_loss = 0.0

        train_writer.add_summary(summary, i)
        if (i + 1) % lr_rate == 0:
            learning_rate = learning_rate * lr_multiplier

        # TIMEOUT:
        if i > top_loss_step + timeout_batches:
            print("No progress in last %s steps, stopping training" %timeout_batches)
            break
        print(loss_, learning_rate)
    return
