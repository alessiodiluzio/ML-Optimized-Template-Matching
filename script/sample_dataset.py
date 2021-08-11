import shutil
import os
import sys


def make_samples(data_size, source, dest):
    for r, _, images in os.walk(source):
        for i in range(data_size):
            img = images[i]
            shutil.copyfile(os.path.join(r, img), os.path.join(dest, img))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 3:
        print('Usage python sample_dataset.py <n_samples> <data_source> <data_dest>')
        exit(0)
    try:
        sample = int(argv[0])
        data_source = argv[1]
        data_dest = argv[2]
        make_samples(sample, data_source, data_dest)
    except ValueError:
        print('Usage python sample_dataset.py <n_samples>')
        exit(0)
