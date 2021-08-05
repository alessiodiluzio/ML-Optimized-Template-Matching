import tensorflow as tf


@tf.function
def true_positives(predictions, labels):
    return tf.math.count_nonzero(predictions * labels, dtype=tf.float32)


@tf.function
def true_negatives(predictions, labels):
    return tf.math.count_nonzero((predictions - 1) * (labels - 1), dtype=tf.float32)


@tf.function
def false_positives(predictions, labels):
    return tf.math.count_nonzero(predictions * (labels - 1), dtype=tf.float32)


@tf.function
def false_negatives(predictions, labels):
    return tf.math.count_nonzero((predictions - 1) * labels, dtype=tf.float32)


@tf.function
def accuracy(logits, labels):

    tp = true_positives(logits, labels)
    tn = true_negatives(logits, labels)
    fp = false_positives(logits, labels)
    fn = false_negatives(logits, labels)

    return tf.cond(tf.equal((tp + tn + fn + fp), 0),
                   lambda: tf.constant(1, dtype=tf.float32), lambda: tf.divide(tf.add(tp, tn), tf.add_n([tp, fn, tn, fp])))


@tf.function
def precision(logits, labels):
    tp = true_positives(logits, labels)
    fp = false_positives(logits, labels)

    return tf.cond(tf.equal(tf.add(tp, fp), tf.constant(0, dtype=tf.float32)),
                   lambda: tf.constant(0, dtype=tf.float32), lambda: tf.divide(tp, tf.add(tp, fp)))


@tf.function
def recall(logits, labels):
    tp = true_positives(logits, labels)
    fn = false_negatives(logits, labels)

    return tf.cond(tf.equal((tf.add(tp, fn)), 0),
                   lambda: tf.constant(0, dtype=tf.float32), lambda: tf.divide(tp, tf.add(tp, fn)))


@tf.function
def f1score(prec, rec):
    return tf.cond(tf.equal((prec + rec), 0), lambda: tf.constant(0, dtype=tf.float32),
                   lambda: tf.multiply(tf.constant(2, dtype=tf.float32),
                                       tf.divide(tf.multiply(prec, rec), tf.add(prec, rec))))
