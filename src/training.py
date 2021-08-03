import tensorflow as tf

from src.metrics import precision, recall, accuracy, f1score
from src.utils import get_balance_factor, get_device

@tf.function
def forward_step(model, inputs, device):
    with tf.device(device):
        output = model(inputs, training=False)
    return output


@tf.function
def forward_backward_step(model, inputs, label, optimizer, loss_fn, balance_factor, device):
    with tf.device(device):
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
    improvement = tf.add(improvement, 1)
    return [new_metric, improvement]


@tf.function
def train(model, training_set, validation_set, train_steps, val_steps, epochs, loss_fn, optimizer, early_stopping=15):
    print("TRACE TRAIN")

    balance_factor = get_balance_factor()
    device = get_device()
    best_loss = tf.constant(1000000, dtype=tf.float32)
    last_improvement = tf.constant(0, dtype=tf.float32)
    early_stopping = tf.constant(early_stopping, tf.float32)

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
        print('TRACE TRAIN LOOP')
        train_loss_arr = tf.TensorArray(tf.float32, size=train_steps)
        train_f1_score_arr = tf.TensorArray(tf.float32, size=train_steps)
        train_acc_arr = tf.TensorArray(tf.float32, size=train_steps)
        step = tf.constant(0, dtype=tf.int32)
        for image, template, label in training_set:
            logits, loss = forward_backward_step(model, [image, template],
                                                 label, optimizer, loss_fn, balance_factor, device)
            train_loss_arr = train_loss_arr.write(step, loss)
            prec = precision(logits, label)
            rec = recall(logits, label)
            train_f1_score_arr = train_f1_score_arr.write(step, f1score(prec, rec))
            acc = accuracy(logits, label)
            train_acc_arr = train_acc_arr.write(step, acc)
            step = tf.add(step, 1)

        train_loss = tf.reduce_mean(train_loss_arr.stack())
        train_f1_score = tf.reduce_mean(train_f1_score_arr.stack())
        train_accuracy = tf.reduce_mean(train_acc_arr.stack())

        tf.print('\nLoss: ', train_loss, ' F1 Score: ', train_f1_score, ' Accuracy: ', train_accuracy)
        train_loss_history = train_loss_history.write(epoch, train_loss)
        train_f1_score_history = train_f1_score_history.write(epoch, train_f1_score)
        train_accuracy_history = train_accuracy_history.write(epoch, train_accuracy)

        tf.print("\nVALIDATE")
        print('TRACE VAL LOOP')
        val_loss_arr = tf.TensorArray(tf.float32, size=val_steps)
        val_f1_score_arr = tf.TensorArray(tf.float32, size=val_steps)
        val_acc_arr = tf.TensorArray(tf.float32, size=val_steps)
        step = tf.constant(0)
        for image, template, label in validation_set:
            logits = forward_step(model, [image, template], device)
            loss = loss_fn(logits, label, activation=None, balance_factor=balance_factor, training=False)
            val_loss_arr = val_loss_arr.write(step, loss)
            prec = precision(logits, label)
            rec = recall(logits, label)
            val_f1_score_arr = val_f1_score_arr.write(step, f1score(prec, rec))
            acc = accuracy(logits, label)
            val_acc_arr = val_acc_arr.write(step, acc)
            step = tf.add(step, 1)

        val_loss = tf.reduce_mean(val_loss_arr.stack())
        val_f1_score = tf.reduce_mean(val_f1_score_arr.stack())
        val_accuracy = tf.reduce_mean(val_acc_arr.stack())

        tf.print('\nLoss: ', val_loss, ' F1 Score: ', val_f1_score, ' Accuracy: ', val_accuracy)
        val_loss_history = val_loss_history.write(epoch, val_loss)
        val_f1_score_history = val_f1_score_history.write(epoch, val_f1_score)
        val_accuracy_history = val_accuracy_history.write(epoch, val_accuracy)

        best_loss, last_improvement = tf.cond(tf.less(val_loss, best_loss),
                                              lambda: update(val_loss, best_loss, last_improvement, 'Validation Loss'),
                                              lambda: [best_loss, last_improvement])
        epoch = tf.cond(tf.greater_equal(last_improvement, early_stopping), lambda: epochs, lambda: epoch)

    return train_loss_history.stack(), train_f1_score_history.stack(), train_accuracy_history.stack(), \
           val_loss_history.stack(), val_f1_score_history.stack(), val_accuracy_history.stack()
