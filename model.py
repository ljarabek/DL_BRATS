import tensorflow as tf
import testing

A = tf.placeholder(dtype = tf.float32)
B = tf.placeholder(dtype = tf.float32)
#C = A+B
#sess = tf.Session()
#sess.run(C, feed_dict={A:5.0, B:3.2})


_a = testing.outputToChannels(5).flatten(0)
_b = testing.outputToChannels(25).flatten(0)
print(_a.shape)

def l2_norm(a):
    return (tf.nn.l2_loss(a) ** 1/2)

"""A = tf.placeholder(dtype = tf.float32)
B = tf.placeholder(dtype = tf.float32)
PT_2 = l2_norm(tf.multiply(A,B))
P2_2 = l2_norm(A) ** 2
T2_2 = l2_norm(B) ** 2
Jacc = tf.divide(PT_2, P2_2 + T2_2 - PT_2)
"""
labels = tf.placeholder(shape = _a.shape, dtype = tf.float32)
predictions = tf.placeholder(shape = _b.shape, dtype = tf.float32)

#labels = tf.contrib.layers.flatten(labels)
#predictions = tf.contrib.layers.flatten(predictions)
truepos = tf.reduce_sum(labels * predictions)
falsepos = tf.reduce_sum(predictions) - truepos
falseneg = tf.reduce_sum(labels) - truepos
jaccard = (truepos ) / (truepos + falseneg + falsepos)
Loss = -tf.log(jaccard)

"""def loss(a,b):

    return (1-Jacc)"""
#x = loss(A,B)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    testing = sess.run(jaccard, feed_dict={labels: _a, predictions: _b})
print(testing)

