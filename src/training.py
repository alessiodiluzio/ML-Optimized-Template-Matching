import tensorflow as tf

from src.metrics import precision, recall, accuracy, f1score


@tf.function
def forward_step(model, inputs):
    output = model(inputs, training=False)
    return output


@tf.function
def forward_backward_step(model, inputs, label, loss_fn, loss_balance_factor, optimizer):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss = loss_fn(logits, label, activation=None, balance_factor=loss_balance_factor, training=True)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return logits, loss


@tf.function
def update(best_metric, new_metric, metric_name):
    tf.print('Improve ', metric_name, ' value: ', best_metric, ' ----> ', new_metric)
    return [new_metric, tf.constant(0)]


@tf.function
def train(model, training_set, validation_set, train_steps, val_steps, epochs, optimizer, loss_fn, loss_balance_factor,
          early_stopping):

    train_loss_history = tf.TensorArray(tf.float32, size=epochs)
    train_f1_score_history = tf.TensorArray(tf.float32, size=epochs)
    train_accuracy_history = tf.TensorArray(tf.float32, size=epochs)

    val_loss_history = tf.TensorArray(tf.float32, size=epochs)
    val_f1_score_history = tf.TensorArray(tf.float32, size=epochs)
    val_accuracy_history = tf.TensorArray(tf.float32, size=epochs)

    train_loss_arr = tf.TensorArray(tf.float32, size=train_steps)
    train_f1_score_arr = tf.TensorArray(tf.float32, size=train_steps)
    train_acc_arr = tf.TensorArray(tf.float32, size=train_steps)

    val_loss_arr = tf.TensorArray(tf.float32, size=val_steps)
    val_f1_score_arr = tf.TensorArray(tf.float32, size=val_steps)
    val_acc_arr = tf.TensorArray(tf.float32, size=val_steps)

    best_metric = tf.constant(1000000.)
    last_improvement = tf.constant(0)

    pretty_line = '\n! --------------------------------------------------------- !\n'
    for epoch in tf.range(epochs):
        tf.print(pretty_line, 'Epoch: ', tf.add(epoch, 1), '/', epochs)

        tf.print('\nTRAIN')
        print('TRACE TRAIN')

        step = tf.constant(0, dtype=tf.int32)
        for image, template, label in training_set:
            tf.print('\rStep ', tf.add(step, 1), '/', train_steps, end='')
            logits, loss = forward_backward_step(model, [image, template], label,
                                                 loss_fn, loss_balance_factor, optimizer)
            step = tf.cast(step, dtype=tf.int32)
            train_loss_arr = train_loss_arr.write(step, loss)
            prec = precision(logits, label)
            rec = recall(logits, label)
            train_f1_score_arr = train_f1_score_arr.write(step, f1score(prec, rec))
            acc = accuracy(logits, label)
            train_acc_arr = train_acc_arr.write(step, acc)
            step = tf.add(step, 1)

        train_loss = tf.reduce_mean(train_loss_arr.stack())
        train_f1_score = tf.reduce_mean(train_f1_score_arr.stack())
        train_acc = tf.reduce_mean(train_acc_arr.stack())
        tf.print('\nLoss: ', train_loss, ' F1 Score: ', train_f1_score, ' Accuracy: ', train_acc)

        train_loss_history = train_loss_history.write(epoch, train_loss)
        train_f1_score_history = train_f1_score_history.write(epoch, train_f1_score)
        train_accuracy_history = train_accuracy_history.write(epoch, train_acc)

        tf.print("\nVALIDATE")
        print('TRACE VALIDATE')

        step = tf.constant(0, dtype=tf.int32)
        for image, template, label in validation_set:
            tf.print('\rStep ', tf.add(step, 1), '/', val_steps, end='')
            logits = forward_step(model, [image, template])
            loss = loss_fn(logits, label, activation=None, balance_factor=loss_balance_factor, training=False)
            val_loss_arr = val_loss_arr.write(step, loss)
            prec = precision(logits, label)
            rec = recall(logits, label)
            val_f1_score_arr = val_f1_score_arr.write(step, f1score(prec, rec))
            acc = accuracy(logits, label)
            val_acc_arr = val_acc_arr.write(step, acc)
            step = tf.add(step, 1)
        val_loss = tf.reduce_mean(val_loss_arr.stack())
        val_f1_score = tf.reduce_mean(val_f1_score_arr.stack())
        val_acc = tf.reduce_mean(val_acc_arr.stack())
        tf.print('\nLoss: ', val_loss, ' F1 Score: ', val_f1_score, ' Accuracy: ', val_acc)

        val_loss_history = val_loss_history.write(epoch, val_loss)
        val_f1_score_history = val_f1_score_history.write(epoch, val_f1_score)
        val_accuracy_history = val_accuracy_history.write(epoch, val_acc)

        best_metric, last_improvement = tf.cond(tf.less(val_loss, best_metric),
                                                 lambda: update(best_metric, val_loss,
                                                                'Validation Loss'),
                                                 lambda: [best_metric, tf.add(last_improvement, 1)])

        epoch = tf.cond(tf.greater_equal(last_improvement, early_stopping),
                        lambda: epochs, lambda: epoch)
        tf.print('Best Metric ', best_metric, 'Last improvement ', last_improvement)

    return train_loss_history.stack(), train_f1_score_history.stack(), train_accuracy_history.stack(), \
            val_loss_history.stack(), val_f1_score_history.stack(), val_f1_score_history.stack(), \
            val_accuracy_history.stack()
