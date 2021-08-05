import sys
import tensorflow as tf

from src.config import Config
from src import config
from src.run import run_train, run_test

message = 'Usage: python main.py <mode> [-datapath=<data_path>] [-eager] [-epochs=<n epochs>]\n' \
          'mode:\n' \
          '\ttrain to run training.\n' \
          '\ttest to run prediction.\n' \
          '-datapath train/test performed on images contained at data_path.\n' \
          '-eager for eager execution\n' \
          '-epochs for set the number of epochs'


def main(_):
    argv = sys.argv[1:]
    mode = None
    eager = False
    epochs = None
    data_path = None
    for arg in argv:
        if arg == 'train':
            mode = 'train'
        elif arg == 'test':
            mode = 'test'
        elif arg == '-eager':
            eager = True
        elif '-epochs' in arg:
            epochs = int(arg.split('=')[1].strip())
        elif '-datapath' in arg:
            data_path = arg.split('-datapath')[1].strip()

    configuration = Config(epochs=epochs, data_path=data_path)

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
        # run_test()


if __name__ == "__main__":
    print(f'Running on platform: {config.OS}')
    tf.compat.v1.app.run()
