import tensorflow as tf
from src.utils import get_zero_base_label


@tf.function
def get_balanced_weigths(balance_factor, label):
    label_true = tf.divide(tf.add(label, 1), 2)
    label_false = tf.multiply(tf.divide(tf.add(label, - 1), 2), -1)
    return tf.add(tf.multiply(tf.add(1.0, -balance_factor), label_true), tf.multiply(balance_factor, label_false))


@tf.function
def cross_entropy_loss(logits, labels, activation, balance_factor, training=True):
    label = get_zero_base_label(labels)
    cross_entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
    if activation.lower() == 'sigmoid':
        cross_entropy = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits)
    if training:
        cross_entropy = tf.expand_dims(cross_entropy, axis=3)
        weights = get_balanced_weigths(balance_factor, label)
        cross_entropy = tf.math.multiply(cross_entropy, weights)
        return tf.reduce_mean(cross_entropy)
    return cross_entropy


# l(y,v) = log(1 + exp(-yv)
# to avoid overflow when -yv < 0
# log(1 + exp(-yv)) = log(1 + exp(-abs(y,v)) -yv + max(0, yv)
@tf.function
def compute_logistic_loss(labels, logits):
    x = tf.math.multiply(logits, labels)
    loss = tf.math.log(1 + tf.math.exp(-1 * tf.math.abs(x))) - x + tf.math.maximum(tf.zeros(x.shape), x)
    return loss


@tf.function
def logistic_loss(logits, label, activation, balance_factor, training=True):
    log_loss = compute_logistic_loss(label, logits)
    if training:
        pass
        weights = get_balanced_weigths(balance_factor, label)
        log_loss = tf.math.multiply(log_loss, weights)
    log_loss = tf.reduce_mean(log_loss)
    return log_loss
