import platform


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
BATCH_SIZE = 5
EPOCHS = 50
LEARNING_RATE = 0.0000099

# UTILS
X_1 = 0
Y_1 = 1
X_2 = 2
Y_2 = 3


class Config:
    def __init__(self, epochs=None, data_path=None):
        self.__epochs = epochs
        self. __data_path = data_path
        if self.__epochs is None:
            self.__epochs = EPOCHS
        if self.__data_path is None:
            self.__data_path = DATA_PATH

    def get_epochs(self):
        return self.__epochs

    def get_data_path(self):
        return self.__data_path
