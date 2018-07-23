import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
from trainingUtils import *
from postprocessing import channelsToOutput, saveSegmentation
import dictionary
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import random as rand
from tqdm import tqdm
import logging
import os
from loss_function import *
from layers import *
from preprocessing import *
from time import time

time = time()
cp = ConfigParser()
cp.read("config.ini")
cfg = cp['CROP']
# val_size = int(cfg[''])
for d in cfg:
    # print(cfg[d])
    exec("%s = %s" % (d, cfg[d]))
    if d.__contains__('dir'):
        exec("a = %s"%cfg[d])
        if not os.path.exists(a):
            os.makedirs(a)

def makedir(string,noise, save_dir=save_dir):
    dir = save_dir + string + "_" + str(noise) + "_" + str(time)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir + "/"

def train_model(sess, noise, ckpt_dir = ""):
    """

    :param sess:
    :param noise:
    :param ckpt_dir:
    :return: top validation loss and corresponding train loss
    """

    main_dir = makedir("main", noise)
    if ckpt_dir == "":
        checkpoint_dir = makedir("ckpt", noise, save_dir=main_dir)
    else:
        checkpoint_dir = ckpt_dir
    val_dir = makedir("val", noise, main_dir)
    train_dir = makedir("train", noise, main_dir)
    npy_dir = makedir("npy", noise, main_dir)
    test_dir = makedir("test", noise, main_dir)
    logging.basicConfig(filename = main_dir + "logg.log", level=logging.DEBUG)
    _input, _answer = getBatchTraining()
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())  # initialize variables
    train_writer = tf.summary.FileWriter(train_dir, sess.graph)  # write graph to tensorboard
    val_writer = tf.summary.FileWriter(val_dir, sess.graph)  # write graph to tensorboard
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
        _input = _input + np.random.normal(scale=noise, size=_input.shape)
        #COMPUTE LOSS AND TRAIN MODEL:
        otpt, loss_, summary, _ = sess.run(["output:0", "loss:0", merged, "train"],
                                           feed_dict={"input:0": _input, "phase_train:0": True, "answer:0": _answer,
                                                      "learning_rate:0": learning_rate})
        loss_ = -loss_
        logging.info("loss: %s learning_rate = %s"%(str(loss_), learning_rate)) # log loss
        if i % save_losses_every == 0:  # save losses and compute validation loss
            for dfdskjfmsdfm in range(val_size):
                _input, _answer = getBatchVal()

                otpt, summary_val, v_loss = sess.run(["output:0", merged, "loss:0"],
                                         feed_dict={"input:0": _input, "phase_train:0": False, "answer:0": _answer})
                v_loss = -v_loss
                val_loss += v_loss/float(val_size)

            losses.append([loss_, val_loss])

            val_writer.add_summary(summary_val, i//save_losses_every)
            print(losses)
            np.save(npy_dir + "train_val_loss.npy", arr=losses)
            print("validation loss: %s , training loss: %s" % (val_loss, loss_))



            if i > top_loss_b and val_loss<val_loss_top:    # at best validation loss:
                top_loss = loss_
                val_loss_top = val_loss
                print("saving top results, train loss: %s validation loss: %s" % (top_loss, val_loss_top))
                top_loss_step = i
                save(saver, sess, checkpoint_dir, i)        # save model at best loss
                np.save(npy_dir + "train_val_best_loss.npy", arr=[loss_,val_loss])

                _input, ID = getBatchTest(True)
                otpt = sess.run(["output:0"], feed_dict={"input:0": _input, "phase_train:0": False,})
                otpt = channelsToOutput(otpt)
                saveSegmentation(_input[0, 2], otpt, i, test_dir)

            val_loss = 0.0

        train_writer.add_summary(summary, i)
        if (i + 1) % lr_rate == 0:
            learning_rate = learning_rate * lr_multiplier

        # TIMEOUT:
        if i > top_loss_step + timeout_batches:
            print("No progress in last %s steps, stopping training" %timeout_batches)
            return val_loss_top, top_loss
            #break
        print(loss_, learning_rate , i)
    return
