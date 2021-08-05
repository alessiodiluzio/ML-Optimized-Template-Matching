import time

import tensorflow as tf
import os

from src import config
from src.training import Trainer
from src.test import test
from src.model import Siamese
from src.loss import logistic_loss
from src.utils import plot_metrics, plot, get_loss_balance_factor, get_device
from src.dataset import get_dataset


def run_train(configuration):
    model = Siamese()

    loss_balance_factor = get_loss_balance_factor()
    device = get_device()
    training_set, validation_set, train_steps, val_steps = get_dataset(configuration.get_data_path(), config.BATCH_SIZE,
                                                                       show=False)
    trainer = Trainer(model, training_set, validation_set, train_steps, val_steps, configuration.get_epochs(),
                      tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE), logistic_loss, loss_balance_factor,
                      device, 15)

    start = time.time()
    trainer()
    print(f'Elapsed {time.time() - start}')
    history = trainer.get_training_data()

    model_history = {
        'train_loss': history[0],
        'train_acc': history[1],
        'train_f1score': history[2],
        'val_loss': history[3],
        'val_acc': history[4],
        'val_f1score': history[5]
    }

    # plot_metrics(model_history, 'plot')

    _, validation_set, _, _ = get_dataset()
    dest_path = 'image'
    for i, (image, template, label) in zip(range(3), validation_set.take(3)):
        prediction = model([image, template], training=False)
        filename = f'prediction_{i}_model_{model.name}.jpg'
        file_path = os.path.join(dest_path, filename)
        plot(image[[0]], template[0], label[0], prediction[0], target='save', dest=file_path)


def run_test(test_path=config.DATA_PATH):
    # training_set, validation_set, train_step, val_step =
    # get_dataset(data_path=test_path, batch_size=1, split_perc=1, show=False)
    test(test_set=[], output_path='image')
