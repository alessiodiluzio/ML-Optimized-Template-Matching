import tensorflow as tf

from src.metrics import precision, recall, accuracy, f1score
from src.utils import get_balance_factor, get_device


@tf.function
def forward_step(model, inputs):
    output = model(inputs, training=False)
    return output


@tf.function
def forward_backward_step(model, inputs, label, optimizer, loss_fn, balance_factor):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss = loss_fn(logits, label, activation=None, balance_factor=balance_factor, training=True)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return logits, loss


@tf.function
def update(new_metric, old_metric, improvement, metric_name):
    print('TRACE UPDATE')
    tf.print('Improve ', metric_name, ' value: ', old_metric, ' ----> ', new_metric)
    old_metric.assign(new_metric)
    improvement.assign_add(improvement, 1)
    return 0
    # return [new_metric, improvement]


@tf.function
def train_loop(model, training_set, train_steps, optimizer, loss_fn, balance_factor):
    print('TRACE TRAIN LOOP')
    train_loss = tf.TensorArray(tf.float32, size=train_steps)
    train_f1_score = tf.TensorArray(tf.float32, size=train_steps)
    train_acc = tf.TensorArray(tf.float32, size=train_steps)
    for step, (image, template, label) in enumerate(training_set):
        logits, loss = forward_backward_step(model, [image, template],
                                             label, optimizer, loss_fn, balance_factor)
        step = tf.cast(step, dtype=tf.int32)
        train_loss = train_loss.write(step, loss)
        prec = precision(logits, label)
        rec = recall(logits, label)
        train_f1_score = train_f1_score.write(step, f1score(prec, rec))
        acc = accuracy(logits, label)
        train_acc = train_acc.write(step, acc)
        step = tf.add(step, 1)
    train_loss = tf.reduce_mean(train_loss.stack())
    train_f1_score = tf.reduce_mean(train_f1_score.stack())
    train_acc = tf.reduce_mean(train_acc.stack())
    return train_loss, train_f1_score, train_acc


@tf.function
def val_loop(model, validation_set, val_steps, loss_fn, balance_factor):
    print('TRACE VAL LOOP')
    val_loss = tf.TensorArray(tf.float32, size=val_steps)
    val_f1_score = tf.TensorArray(tf.float32, size=val_steps)
    val_acc = tf.TensorArray(tf.float32, size=val_steps)
    # step = tf.Variable(0)
    for step, (image, template, label) in enumerate(validation_set):
        step = tf.cast(step, dtype=tf.int32)
        logits = forward_step(model, [image, template])
        loss = loss_fn(logits, label, activation=None, balance_factor=balance_factor, training=False)
        val_loss = val_loss.write(step, loss)
        prec = precision(logits, label)
        rec = recall(logits, label)
        val_f1_score = val_f1_score.write(step, f1score(prec, rec))
        acc = accuracy(logits, label)
        val_acc = val_acc.write(step, acc)
        step = tf.add(step, 1)
    val_loss = tf.reduce_mean(val_loss.stack())
    val_f1_score = tf.reduce_mean(val_f1_score.stack())
    val_acc = tf.reduce_mean(val_acc.stack())
    return val_loss, val_f1_score, val_acc


@tf.function
def train(model, training_set, validation_set, train_steps, val_steps, epochs, loss_fn, optimizer, early_stopping,
          best_loss, last_improvement):
    print("TRACE TRAIN")

    balance_factor = get_balance_factor()
    # device = get_device()
    train_loss_history = tf.TensorArray(tf.float32, size=epochs)
    train_f1_score_history = tf.TensorArray(tf.float32, size=epochs)
    train_accuracy_history = tf.TensorArray(tf.float32, size=epochs)

    val_loss_history = tf.TensorArray(tf.float32, size=epochs)
    val_f1_score_history = tf.TensorArray(tf.float32, size=epochs)
    val_accuracy_history = tf.TensorArray(tf.float32, size=epochs)

    pretty_line = '\n! --------------------------------------------------------- !\n'
    for epoch in tf.range(epochs):
        print('Trace Epoch')
        tf.print(pretty_line, 'Epoch: ', tf.add(epoch, 1), '/', epochs)

        tf.print('\nTRAIN')
        train_loss, train_f1_score, train_acc = train_loop(model, training_set, train_steps, optimizer, loss_fn,
                                                           balance_factor)
        tf.print('\nLoss: ', train_loss, ' F1 Score: ', train_f1_score, ' Accuracy: ', train_acc)

        train_loss_history = train_loss_history.write(epoch, train_loss)
        train_f1_score_history = train_f1_score_history.write(epoch, train_f1_score)
        train_accuracy_history = train_accuracy_history.write(epoch, train_acc)

        tf.print("\nVALIDATE")
        val_loss, val_f1_score, val_acc = val_loop(model, validation_set, val_steps, loss_fn, balance_factor)

        tf.print('\nLoss: ', val_loss, ' F1 Score: ', val_f1_score, ' Accuracy: ', val_acc)

        val_loss_history = val_loss_history.write(epoch, val_loss)
        val_f1_score_history = val_f1_score_history.write(epoch, val_f1_score)
        val_accuracy_history = val_accuracy_history.write(epoch, val_acc)

        last_improvement.assign(tf.cond(tf.less(val_loss, best_loss),
                                        lambda: update(val_loss, best_loss, last_improvement, 'Validation Loss'),
                                        lambda: tf.add(last_improvement, 1)))
        epoch = tf.cond(tf.greater_equal(last_improvement, early_stopping), lambda: epochs, lambda: epoch)

    return train_loss_history.stack(), train_f1_score_history.stack(), train_accuracy_history.stack(), \
           val_loss_history.stack(), val_f1_score_history.stack(), val_accuracy_history.stack()
