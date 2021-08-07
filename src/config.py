import platform
import tensorflow as tf

OS = platform.system()

# PATH
DATA_PATH = 'samples'

# PREPROCESSING
IMAGE_DIM = 255
IMAGE_OUTPUT_DIM = 22
CROP_OUTPUT_DIM = 6
OUTPUT_CHANNELS = 128
OUTPUT_DIM = 17

CHANNELS = 3
CROP_SIZE = 127
CROP_BOX = (CROP_SIZE, CROP_SIZE)
NUM_BOXES = 1

# TRAINING
BATCH_SIZE = 10
EPOCHS = 50
LEARNING_RATE = 0.0000099

# UTILS
X_1 = 0
Y_1 = 1
X_2 = 2
Y_2 = 3


class Config:
    def __init__(self, data_path, epochs, batch_size, learning_rate, optimizer):
        self.__epochs = epochs
        self. __data_path = data_path
        self.__batch_size = batch_size
        self.__optimizer_name = optimizer
        self.__optimizer = None
        self.__learning_rate = learning_rate
        if self.__epochs is None:
            self.__epochs = EPOCHS
        if self.__data_path is None:
            self.__data_path = DATA_PATH
        if self.__batch_size is None:
            self.__batch_size = BATCH_SIZE
        if self.__learning_rate is None:
            self.__learning_rate = LEARNING_RATE
        if self.__optimizer_name is None or self.__optimizer_name.lower() == 'adam':
            self.__optimizer = tf.keras.optimizers.Adam(learning_rate=self.__learning_rate)
            self.__optimizer_name = 'Adam'
        elif self.__optimizer_name.lower() == 'sgd':
            self.__optimizer = tf.keras.optimizers.SGD(learning_rate=self.__learning_rate)
        elif self.__optimizer_name.lower() == 'rmsprop':
            self.__optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.__learning_rate)
        elif self.__optimizer_name.lower() == 'adagrad':
            self.__optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.__learning_rate)
        elif self.__optimizer_name.lower() == 'adamax':
            self.__optimizer = tf.keras.optimizers.Adamax(learning_rate=self.__learning_rate)
        elif self.__optimizer_name.lower() == 'adadelta':
            self.__optimizer = tf.keras.optimizers.Adadelta(learning_rate=self.__learning_rate)
        else:
            self.__optimizer = tf.keras.optimizers.Adam(learning_rate=self.__learning_rate)
            self.__optimizer_name = 'Adam'

    def get_epochs(self):
        return self.__epochs

    def get_data_path(self):
        return self.__data_path

    def get_batch_size(self):
        return self.__batch_size

    def get_optimizer_name(self):
        return self.__optimizer_name.upper()

    def get_optimizer(self):
        return self.__optimizer

    def get_learning_rate(self):
        return self.__learning_rate
