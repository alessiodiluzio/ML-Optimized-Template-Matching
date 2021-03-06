import time

import tensorflow as tf
import os

from src.training import Trainer
from src.test import test
from src.model import Siamese
from src.loss import logistic_loss, cross_entropy_loss
from src.utils import plot_metrics, plot, get_loss_balance_factor, get_device
from src.dataset import get_train_set


def run_train(configuration):
    model = Siamese()

    msg_line = '# -------------------------------------------\n'
    msg_line_1 = '# --- TRAIN\n'
    msg_line_2 = f'# --- Net {model.name}\n'
    msg_line_3 = f'# --- Epochs {configuration.get_epochs()}\n'
    msg_line_4 = f'# --- Batch size {configuration.get_batch_size()}\n'
    msg_line_5 = f'# --- Optimizer {configuration.get_optimizer_name()}\n'
    msg_line_6 = f'# --- Learning Rate {configuration.get_learning_rate()}\n'

    msg = msg_line + msg_line_1 + msg_line_2 + msg_line_3 + msg_line_4 + msg_line_5 + msg_line_6 + msg_line

    print(msg)

    loss_balance_factor = get_loss_balance_factor()
    device = get_device()
    training_set, validation_set, train_steps, val_steps = get_train_set(configuration.get_data_path(),
                                                                         configuration.get_batch_size(), show=False)

    trainer = Trainer(model, training_set, validation_set, train_steps, val_steps, configuration.get_epochs(),
                      configuration.get_optimizer(), logistic_loss, loss_balance_factor,
                      device, 15, 'saved_model', 'checkpoint')

    start = time.time()
    history = trainer()
    print(f'\nElapsed {time.time() - start}')

    model_history = {
        'train_loss': history[0],
        # 'train_f1score': history[1],
        # 'train_acc': history[2],
        'val_loss': history[1],
        # 'val_f1score': history[4],
        # ' val_acc': history[5],
    }

    plot_metrics(model_history, 'plot')

    best_model = tf.keras.models.load_model('saved_model')

    _, validation_set, _, _ = get_train_set(configuration.get_data_path(), configuration.get_batch_size())
    dest_path = 'image'
    for i, (image, template, label) in zip(range(3), validation_set.take(3)):
        prediction = best_model([image, template], training=False)
        filename = f'prediction_{i}_model_{best_model.name}.jpg'
        file_path = os.path.join(dest_path, filename)
        plot(image[[0]], template[0], label[0], prediction[0], target='save', dest=file_path)


def run_test(configuration):
    test(configuration.get_data_path())
