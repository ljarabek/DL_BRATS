import tensorflow as tf
import numpy as np

def jaccard_loss(P,T):
    """jaccard_loss function recieves 2 arguments P-predictions, T- true values
     and returns a value 1 - Jaccard index"""

    #power stand for |.| (moc mnozice)
    #PT stands for the element wise product of matrix P and T
    PT = tf.norm(tf.multiply(P,T), ord=2)
    P = tf.norm(P, ord=2)
    T = tf.norm(T, ord=2)

    return 1 - tf.divide(PT, P**2 + T ** 2 - PT)


#####################################################3
#### test
##Dice in Jaccard

def dice(a,b):
    sumAB = np.sum(a*b)
    sumA = np.sum(a)
    sumB = np.sum(b)
    return 2*sumAB/(sumA+sumB)

def jaccard(a,b):
    sumAB = np.sum(a * b) #to naredi produkt po elemtih
    sumA = np.sum(a)
    sumB = np.sum(b)
    return (sumAB)/(sumA + sumB - sumAB)

def jaccard2(P, T):
    #fukncija izgube kot je v članku
    normPT = np.linalg.norm(P * T)
    normP = np.linalg.norm(P)
    normT = np.linalg.norm(T)

    return normPT/(normP**2 + normT**2 - normPT)


def jaccard_coef_logloss(labels, predictions, smooth=1e-10):
    """ Loss function based on jaccard coefficient.

    Parameters
    ----------
    labels : tf.Tensor
        tensor containing target mask.
    predictions : tf.Tensor
        tensor containing predicted mask.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    tf.Tensor
        tensor containing negative logarithm of jaccard coefficient.
    """
    labels = tf.contrib.layers.flatten(labels)
    predictions = tf.contrib.layers.flatten(predictions)
    truepos = tf.reduce_sum(labels * predictions)
    falsepos = tf.reduce_sum(predictions) - truepos
    falseneg = tf.reduce_sum(labels) - truepos
    jaccard = tf.divide((truepos + smooth) , (smooth + truepos + falseneg + falsepos))
    return -tf.log(jaccard + smooth)
    #return 1 - jaccard

