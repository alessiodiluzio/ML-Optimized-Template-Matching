import tensorflow as tf


@tf.function
def true_positives(predictions, labels):
    """
    :param predictions: Current batch predictions
    :param labels: Current batch labels
    :return: Number of True Positive predictions for the current batch
    """
    return tf.math.count_nonzero(predictions * labels, dtype=tf.float32)


@tf.function
def true_negatives(predictions, labels):
    """
    :param predictions: Current batch predictions
    :param labels: Current batch labels
    :return: Number of True Negative predictions for the current batch
    """
    return tf.math.count_nonzero((predictions - 1) * (labels - 1), dtype=tf.float32)


@tf.function
def false_positives(predictions, labels):
    """
    :param predictions: Current batch predictions
    :param labels: Current batch labels
    :return: Number of False Positive predictions for the current batch
    """
    return tf.math.count_nonzero(predictions * (labels - 1), dtype=tf.float32)


@tf.function
def false_negatives(predictions, labels):
    """
    :param predictions: Current batch predictions
    :param labels: Current batch labels
    :return: Number of False Negative predictions for the current batch
    """
    return tf.math.count_nonzero((predictions - 1) * labels, dtype=tf.float32)


@tf.function
def accuracy(predictions, labels):
    """
    Accuracy = (TP + TN)/(TP + TN + FP + FN)
    :param predictions: Current batch predictions
    :param labels: Current batch predictions
    :return: Accuracy for the current batch.
    """
    tp = true_positives(predictions, labels)
    tn = true_negatives(predictions, labels)
    fp = false_positives(predictions, labels)
    fn = false_negatives(predictions, labels)

    return tf.cond(tf.equal((tp + tn + fn + fp), 0),
                   lambda: tf.constant(1, dtype=tf.float32), lambda: tf.divide(tf.add(tp, tn), tf.add_n([tp, fn, tn, fp])))


@tf.function
def precision(predictions, labels):
    """
    Precision = (TP)/(TP + FP)
    :param predictions: Current batch predictions
    :param labels: Current batch predictions
    :return: Precision for the current batch.
    """
    tp = true_positives(predictions, labels)
    fp = false_positives(predictions, labels)

    return tf.cond(tf.equal(tf.add(tp, fp), tf.constant(0, dtype=tf.float32)),
                   lambda: tf.constant(0, dtype=tf.float32), lambda: tf.divide(tp, tf.add(tp, fp)))


@tf.function
def recall(predictions, labels):
    """
    Recall = (TP)/(TP + FN)
    :param predictions: Current batch predictions
    :param labels: Current batch predictions
    :return: Recall for the current batch.
    """
    tp = true_positives(predictions, labels)
    fn = false_negatives(predictions, labels)

    return tf.cond(tf.equal((tf.add(tp, fn)), 0),
                   lambda: tf.constant(0, dtype=tf.float32), lambda: tf.divide(tp, tf.add(tp, fn)))


@tf.function
def f1score(prec, rec):
    """
    F1 = 2 * (Precision * Recall)/(Precision + Recall)
    :param prec: Current batch precision
    :param rec: Current batch recall
    :return: F1 score for the current batch.
    """
    return tf.cond(tf.equal((prec + rec), 0), lambda: tf.constant(0, dtype=tf.float32),
                   lambda: tf.multiply(tf.constant(2, dtype=tf.float32),
                                       tf.divide(tf.multiply(prec, rec), tf.add(prec, rec))))
