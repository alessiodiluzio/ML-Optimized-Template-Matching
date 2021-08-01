import tensorflow as tf

from src import EPOCHS, DATA_PATH, LEARNING_RATE, BATCH_SIZE
from src.training import train
from src.test import test
from src.model import Siamese
from src.loss import logistic_loss, cross_entropy_loss


def run_train():
    model = Siamese()
    train(model, DATA_PATH, EPOCHS, BATCH_SIZE, plot_path='plot', image_path='image', loss_fn=logistic_loss,
          optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), early_stopping=15,
          plot_val_logits=True)


def run_test(test_path=DATA_PATH):
    #training_set, validation_set, train_step, val_step = get_dataset(data_path=test_path, batch_size=1, split_perc=1, show=False)
    test(test_set=[], output_path='image')
