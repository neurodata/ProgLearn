import math
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa


def logDiff(yTrue, yPred):
    return K.sum(K.log(yTrue) - K.log(yPred))


# sum over samples in batch (anchors) ->
# average over similar samples (positive) ->
# of - log softmax positive / sum negatives (wrt cos similarity)
# i.e. \sum_i -1/|P(i)| \sum_{p \in P(i)} log [exp(z_i @ z_p / t) / \sum_{n \in N(i)} exp(z_i @ z_n / t)]
# = \sum_i [log[\sum_{n \in N(i)} exp(z_i @ z_n / t)] - 1/|P(i)| \sum_{p \in P(i)} log [exp(z_i @ z_p / t)]]
def supervised_contrastive_loss(yTrue, yPred):
    print(yTrue)
    print(yPred)
    temp = 0.1
    # r = tf.reshape(yPred,(32,10))
    # y = tf.reshape(yTrue,(32,10))
    r = yPred
    y = yTrue

    r, _ = tf.linalg.normalize(r, axis=1)
    r_dists = tf.matmul(r, tf.transpose(r))
    print(r_dists.shape)
    """
  for i in range(len(r_dists)):
    r_dists[i] = 0
  """
    # r_dists = tf.linalg.set_diag(r_dists, tf.zeros(r_dists.shape[0], dtype=r_dists.dtype))  # exclude itself distance
    r_dists = r_dists / temp
    y_norms = tf.reduce_sum(y * y, 1)
    y = y_norms - 2 * tf.matmul(y, tf.transpose(y)) + tf.transpose(y_norms)

    y = tf.cast(y / 2, r_dists.dtype)  # scale onehot distances to 0 and 1
    negative_sum = tf.math.log(
        tf.reduce_sum(y * tf.exp(r_dists), axis=1)
    )  # y zeros diagonal 1's
    positive_sum = (1 - y) * r_dists

    n_nonzero = tf.math.reduce_sum(1 - y, axis=1) - 1  # Subtract diagonal
    positive_sum = tf.reduce_sum(positive_sum, axis=1) / tf.cast(
        n_nonzero, positive_sum.dtype
    )
    loss = tf.reduce_sum(negative_sum - positive_sum)
    return loss


# siamese networks version
def contrastiveLoss(yTrue, yPred):
    print(1 - yPred)
    # make sure the datatypes are the same
    yTrue = tf.cast(yTrue, yPred.dtype)
    squaredPreds = K.square(yPred)
    squaredMargin = K.square(K.maximum(1 - yPred, 0))
    loss = K.mean(yTrue * squaredPreds + (1 - yTrue) * squaredMargin)
    print(loss, type(loss))
    return loss


def cosSimilarity(vec1, vec2):
    print(vec1, vec2)
    sim = tf.reduce_sum(tf.reduce_sum(tf.multiply(vec1, vec1)))
    print(sim.shape)
    return sim


def SupervisedContrastiveLoss(yTrue, yPred):
    temp = 0.1
    r = yPred
    y = yTrue
    r, _ = tf.linalg.normalize(r, axis=1)
    r_dists = tf.matmul(r, tf.transpose(r))
    logits = tf.divide(r_dists, temp)
    return tfa.losses.npairs_loss(tf.squeeze(tf.reduce_sum(y * y, 1)), logits)
