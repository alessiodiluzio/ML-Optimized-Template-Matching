""" Implementation of loss  functions """

import tensorflow as tf
from src.utils import get_zero_base_label


@tf.function
def get_balanced_weights(balance_factor, label):
    """
    Build a matrix of weights use to balance dataset due to imbalance between negative and positive pixels.
    The weigths are build against a specified label (Different weight for positive and negative label)
    :param balance_factor: percentage of positive pixels in the entire training set.
    :param label: Label used to build weights matrix
    :return: Weighted label matrix.
    """
    label_true = tf.divide(tf.add(label, 1), 2)
    label_false = tf.multiply(tf.divide(tf.add(label, - 1), 2), -1)
    return tf.add(tf.multiply(tf.add(1.0, -balance_factor), label_true), tf.multiply(balance_factor, label_false))


@tf.function
def cross_entropy_loss(logits, labels, balance_factor, training=True):
    label = get_zero_base_label(labels)
    cross_entropy = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits)
    if training:
        cross_entropy = tf.expand_dims(cross_entropy, axis=3)
        weights = get_balanced_weights(balance_factor, label)
        cross_entropy = tf.math.multiply(tf.squeeze(cross_entropy, axis=-1), weights)
    return tf.reduce_mean(cross_entropy)


@tf.function
def compute_logistic_loss(labels, logits):
    """
    Compute the logistic loss: l(y,v) = log(1 + exp(-yv), to avoid overflow when -yv < 0 the following formula is used
    log(1 + exp(-yv)) = log(1 + exp(-abs(y,v)) -yv + max(0, yv)
    :param labels: Current batch labels
    :param logits: Current batch logits
    :return: Matrix of loss computed for each activation (i.e pixel) of the feature maps in the current batch.
    """
    x = tf.math.multiply(logits, labels)
    loss = tf.math.log(1 + tf.math.exp(-1 * tf.math.abs(x))) - x + tf.math.maximum(tf.zeros(x.shape), x)
    return loss


@tf.function
def logistic_loss(logits, label, balance_factor, training=True):
    """
    Compute the mean logistic loss balanced by a balance factor for the current batch.
    :param logits: Current batch logits
    :param label: Current batch labels
    :param activation: unused
    :param balance_factor: percentage of positive pixels in the entire training set.
    :param training: True if loss must be balanced due to positive/negative imbalance in dataset (training phase)
    false on the contrary (inference/test)
    :return: Mean logistic loss for the current batch.
    """
    log_loss = compute_logistic_loss(label, logits)
    if training:
        weights = get_balanced_weights(balance_factor, label)
        log_loss = tf.math.multiply(log_loss, weights)
    log_loss = tf.reduce_mean(log_loss)
    return log_loss
