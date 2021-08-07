import sys
import tensorflow as tf

from src.config import Config
from src import config
from src.run import run_train, run_test

message = 'Usage: python main.py <mode> [-datapath=<data_path>] [-eager] [-epochs=<n epochs>] [-batch=<batch size>]' \
          '[-optimizer=<optmizer name>] [-lr=<learning rate]\n ' \
          'mode:\n' \
          '\ttrain to run training.\n' \
          '\ttest to run prediction.\n\n' \
          '-datapath train/test performed on images contained at data_path.\n' \
          '-eager for eager execution\n' \
          '-epochs to set the number of epochs\n' \
          '-batch to set the dataset batch size\n' \
          '-optimizer to set the optimizer\n' \
          '-lr to set the learning rate\n'


def main(_):
    argv = sys.argv[1:]
    mode = None
    eager = False
    epochs = None
    data_path = None
    batch_size = None
    optimizer = None
    learning_rate = None
    for arg in argv:
        if arg == 'train':
            mode = 'train'
        elif arg == 'test':
            mode = 'test'
        elif arg == '-eager':
            eager = True
        elif '-epochs' == arg.split('=')[0]:
            epochs = (arg.split('=')[1].strip())
            if epochs.isdigit():
                epochs = int(epochs)
            else:
                epochs = None
        elif '-datapath' == arg.split('=')[0]:
            data_path = arg.split('=')[1].strip()
        elif '-batch' == arg.split('=')[0]:
            batch_size = (arg.split('=')[1].strip())
            if batch_size.isdigit():
                batch_size = int(batch_size)
            else:
                batch_size = None
        elif '-optimizer' == arg.split('=')[0]:
            optimizer = arg.split('=')[1].strip()
        elif '-lr' == arg.split('=')[0]:
            learning_rate = (arg.split('=')[1].strip())
            if learning_rate.isnumeric():
                learning_rate = float(learning_rate)
            else:
                learning_rate = None

    configuration = Config(data_path=data_path, epochs=epochs, batch_size=batch_size,
                           optimizer=optimizer, learning_rate=learning_rate)

    if mode is None:
        print(message)
        exit(0)
    if eager:
        print('Running functions in eager mode...')
        tf.config.run_functions_eagerly(eager)
    if mode == 'train':
        run_train(configuration)
    elif mode == 'test':
        print('TEST MODE work in progress...')
        run_test(configuration)


if __name__ == "__main__":
    print(f'Running on platform: {config.OS}')
    tf.compat.v1.app.run()
