import tensorflow as tf
import os

from src.model import Siamese
from src.metrics import precision, recall, accuracy, f1score
from src.utils import plot, plot_metrics, get_balance_factor, get_device
from src.dataset import get_dataset


@tf.function
def forward_step(model, inputs):
    with tf.device(get_device()):
        output = model(inputs, training=False)
    return output


@tf.function
def forward_backward_step(model, inputs, label, optimizer, loss_fn, balance_factor):
    with tf.device(get_device()):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss = loss_fn(logits, label, activation=None, balance_factor=balance_factor, training=True)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        return logits, loss


def plot_dataset_with_logits(model, dataset, save_path, epoch, prefix=''):
    for i, (image, template, labels) in zip(range(3), dataset.take(3)):
        predictions = model([image, template])
        filename = prefix + '_epoch_{}_sample_{}.jpg'.format(epoch+1, i)
        plot(image[0], template[0], labels[0], predictions[0],
             target='save', dest=os.path.join(save_path, filename))


@tf.function
def train_loop(model, training_set, train_steps, optimizer, loss_fn, balance_factor):
    loss_arr = tf.TensorArray(tf.float32, size=train_steps)
    f1_score_arr = tf.TensorArray(tf.float32, size=train_steps)
    acc_arr = tf.TensorArray(tf.float32, size=train_steps)
    for b, (image, template, label) in zip(range(train_steps), training_set.take(train_steps)):#zip(range(train_steps), training_set.take(train_steps)):
        logits, loss = forward_backward_step(model, [image, template], label, optimizer, loss_fn, balance_factor)
        b = tf.cast(b, dtype=tf.int32)
        loss_arr.write(b, loss)
        prec = precision(logits, label)
        rec = recall(logits, label)
        f1_score_arr.write(b, f1score(prec, rec))
        acc = accuracy(logits, label)
        acc_arr.write(b, acc)
    return loss_arr, f1_score_arr, acc_arr


@tf.function
def val_loop(model, validation_set, val_steps, loss_fn, balance_factor):
    loss_arr = tf.TensorArray(tf.float32, size=val_steps)
    f1_score_arr = tf.TensorArray(tf.float32, size=val_steps)
    acc_arr = tf.TensorArray(tf.float32, size=val_steps)
    for b, (image, template, label) in zip(range(val_steps), validation_set.take(val_steps)):
        logits = forward_step(model, [image, template])
        loss = loss_fn(logits, label, activation=None, balance_factor=balance_factor, training=False)
        b = tf.cast(b, dtype=tf.int32)
        loss_arr.write(b, loss)
        prec = precision(logits, label)
        rec = recall(logits, label)
        f1_score_arr.write(b, f1score(prec, rec))
        acc = accuracy(logits, label)
        acc_arr.write(b, acc)
    return loss_arr, f1_score_arr, acc_arr


def train(model, train_data_path, epochs, batch_size, plot_path,
          image_path, loss_fn, optimizer, early_stopping=None, plot_val_logits=True):

    training_set, validation_set, train_steps, val_steps = get_dataset(train_data_path, batch_size, show=False)

    # Initialize dictionary to store the history
    model.history = {'train_loss': [], 'val_loss': [], 'train_f1score': [],
                     'val_f1score': [], 'train_acc': [], 'val_acc': []}
    balance_factor = get_balance_factor()

    best_loss = 1000000
    last_improvement = 0
    #train_steps = tf.constant(train_steps, dtype=tf.int32)
    #val_steps = tf.constant(val_steps, dtype=tf.int32)

    for epoch in range(epochs):

        print(f'\nEpoch: {epoch+1}/{epochs}')
        print('\n TRAIN')
        train_loss, train_f1_score, train_accuracy = \
            train_loop(model, training_set, train_steps, optimizer, loss_fn, balance_factor)
        train_loss = tf.reduce_mean(train_loss)
        train_f1_score = tf.reduce_mean(train_f1_score)
        train_accuracy = tf.reduce_mean(train_accuracy)
        print(f'Loss: {train_loss} F1 Score: {train_f1_score} Accuracy: {train_accuracy}')

        print("\nVALIDATE")
        val_loss, val_f1_score, val_accuracy = \
            val_loop(model, validation_set, val_steps, loss_fn, balance_factor)
        val_loss = tf.reduce_mean(train_loss)
        val_f1_score = tf.reduce_mean(val_f1_score)
        val_accuracy = tf.reduce_mean(val_accuracy)
        print(f'Loss: {val_loss} F1 Score: {val_f1_score} Accuracy: {val_accuracy}')

        model.history['train_loss'].append(train_loss)
        model.history['train_acc'].append(train_accuracy.result())
        model.history['train_f1score'].append(train_f1_score)

        model.history['val_loss'].append(val_loss)
        model.history['val_acc'].append(val_accuracy)
        model.history['val_f1score'].append(val_f1_score)

        if tf.executing_eagerly():
            if plot_val_logits:
                plot_dataset_with_logits(model, training_set, image_path, epoch, 'train')
                plot_dataset_with_logits(model, validation_set, image_path, epoch, 'val')

            if model.history['val_loss'][-1] < best_loss:
                last_improvement = 0
                model.save_model('checkpoint')
                target_loss = model.history['val_loss'][-1]
                print(f'Model saved. validation loss : {best_loss} --> {target_loss}')
                best_loss = target_loss
            else:
                last_improvement += 1

            if early_stopping is not None and last_improvement >= early_stopping:
                break

    if tf.executing_eagerly():
        plot_metrics(model.history, plot_path)
