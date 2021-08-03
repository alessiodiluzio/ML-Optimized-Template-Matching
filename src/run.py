import tensorflow as tf
import os
import time

from src import EPOCHS, DATA_PATH, LEARNING_RATE, BATCH_SIZE
from src.training import train
from src.test import test
from src.model import Siamese
from src.loss import logistic_loss, cross_entropy_loss
from src.utils import plot_metrics, plot
from src.dataset import get_dataset


def run_train():
    model = Siamese()
    start = time.time()
    training_set, validation_set, train_steps, val_steps = get_dataset(DATA_PATH, BATCH_SIZE, show=False)
    history = train(model, training_set, validation_set, train_steps, val_steps, EPOCHS,
                    loss_fn=logistic_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    early_stopping=15)

    print(f'Elapsed {time.time() - start}')

    model_history = {
        'train_loss': history[0],
        'train_acc': history[1],
        'train_f1score': history[2],
        'val_loss': history[3],
        'val_acc': history[4],
        'val_f1score': history[5]
    }

    plot_metrics(model_history, 'plot')

    _, validation_set, _, _ = get_dataset()
    dest_path = 'image'
    for i, (image, template, label) in zip(range(3), validation_set.take(3)):
        prediction = model([image, template], training=False)
        filename = f'prediction_{i}_model_{model.name}.jpg'
        file_path = os.path.join(dest_path, filename)
        plot(image[[0]], template[0], label[0], prediction[0], target='save', dest=file_path)


def run_test(test_path=DATA_PATH):
    # training_set, validation_set, train_step, val_step =
    # get_dataset(data_path=test_path, batch_size=1, split_perc=1, show=False)
    test(test_set=[], output_path='image')
