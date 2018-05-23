import tensorflow as tf
from testing import *
from loss_function import jaccard_loss

_input, _answer = getBatchTraining()

A = tf.placeholder(dtype = tf.float32)
B = tf.placeholder(dtype = tf.float32)




