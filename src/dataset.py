""" Implementation of Tensorflow input pipeline for Template Matching"""

import tensorflow as tf

from src import config
from src.utils import make_label, get_filenames, plot_dataset


def load_image(filename):
    """
    1)
    Load image from filesystem
    :param filename: path to image
    :return: Tensorflow decoded image.
    """
    raw = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(raw, channels=config.CHANNELS)
    return image


def preprocess(image):
    """
    2)
    Resize and normalize an image, build a random box that will be used in the following crop step
    :param image: Tensorflow decoded image
    :return: image and the coordinates of a box specified by [x_min, x_max, y_min, y_max].
    """
    image = tf.image.resize(image, [config.IMAGE_DIM, config.IMAGE_DIM])
    image /= 255
    return image


def extract_crop(image):
    """
    3)
    Extract crop from an image at specified coordinates.
    :param image: Tensorflow decoded image
    :return: Source image, the crop extracted from source image, the coordinates of the box.
    """
    x1 = tf.random.uniform(shape=[1], minval=0, maxval=config.IMAGE_DIM - config.CROP_SIZE, dtype=tf.int32)
    y1 = tf.random.uniform(shape=[1], minval=0, maxval=config.IMAGE_DIM - config.CROP_SIZE, dtype=tf.int32)

    x1 = tf.cast(x1, dtype=tf.float32)
    y1 = tf.cast(y1, dtype=tf.float32)

    x2 = tf.math.add(x1, config.CROP_SIZE)
    y2 = tf.math.add(y1, config.CROP_SIZE)

    boxes = tf.concat([x1, y1, x2, y2], axis=0)
    boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
    begin = tf.stack([tf.cast(boxes[config.Y_1], dtype=tf.int32), tf.cast(boxes[config.X_1], dtype=tf.int32), 0], axis=0)
    template = tf.slice(image, begin, size=[config.CROP_SIZE, config.CROP_SIZE, 3])
    return image, template, boxes


def generate_ground_truth(image, template, boxes):
    """
    4)
    Generate a ground truth label for the crop extracted from the source image
    :param image: Tensorflow decoded image
    :param template: A crop extracted from source image at coordinates specified by boxes param
    :param boxes: the coordinates of a box specified by [x_min, x_max, y_min, y_max].
    :return: Source image, crop, a 1D image that has the same size of source image and it is composed
                by 'ones' in the pixel that match with crop coordinates, the remaining pixels have value zero
    """
    boxes = tf.cast(boxes, dtype=tf.int32)

    label = make_label(boxes, config.IMAGE_DIM)

    # center_x = tf.divide(tf.add(boxes[2], - boxes[0]), 2)
    # center_y = tf.divide(tf.add(boxes[3], - boxes[1]), 2)
    #
    # center = tf.ones((10, 10))
    # paddings = [[center_y - 4, config.IMAGE_DIM - center_y - 5], [center_x - 4, config.IMAGE_DIM - center_x - 5]]
    # ret = tf.pad(center, paddings, mode='CONSTANT', constant_values=-1)
    # ret = tf.cast(ret, dtype=tf.float32)
    # ret = tf.zeros((config.IMAGE_DIM, config.IMAGE_DIM)) + ret
    # ret = tf.expand_dims(ret, axis=-1)
    return image, template, label


# TODO
def perturb(image, template, label):
    """
    5) (optional)
    Randomly perturb source image and Ground truth label
    :param image: Tensorflow decoded image
    :param template: A crop extracted from source image at coordinates specified by boxes param
    :param label: Ground truth label for crop against source image.
    :return: Source image randomly perturbed, the crop, the ground truth perturbed in the same way of source image
    """
    return image, template, label


def make_train_set(images_path, batch_size, augmentation=False):
    """
    Build a dataset from a list of image
    :param images_path: List containing path of images
    :param batch_size: Hyperparam for net training, the dataset is processed in batch of dim batch_size
    :param augmentation: if true, images are randomly perturbed.
    :return: A dataset ready for a training.
    """
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = dataset.shuffle(len(images_path))
    dataset = dataset.map(load_image)  # 1)
    dataset = dataset.map(preprocess)  # 2)
    dataset = dataset.map(extract_crop)  # 3)
    dataset = dataset.map(generate_ground_truth)  # 4)
    if augmentation:
        pass
        # dataset = dataset.map(perturb)  # 5)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return dataset


def get_train_set(data_path, batch_size, split_perc=0.7, show=False):
    """
    Build training and validation set ready for training phase.
    :param data_path: Path to the folder that contains the dataset images
    :param batch_size: Hyperparam for net training, the dataset is divided in batch of dim batch_size
    :param split_perc: The percentual of dataset that is used for training (the remaining is used as validation set)
    :param show: whether show samples from training set.
    :return: datasets ready for training and validation, training step and validation step
    """
    images = get_filenames(data_path)
    val_index = int(len(images) * split_perc)
    training_images = images[:val_index]
    validation_images = images[val_index:]
    training_set = make_train_set(training_images, batch_size, augmentation=True)
    validation_set = make_train_set(validation_images, batch_size)
    training_step = int(len(training_images)/batch_size)  # training step = | TRAINING_SET |/batch_size
    validation_step = int(len(validation_images)/batch_size)  # validation step = | VALIDATION_SET |/batch_size
    if show:
        plot_dataset(training_set, 3, target='show')
    return training_set, validation_set, training_step, validation_step


def get_test_set(data_path):
    images = get_filenames(data_path)
    test_step = len(images)
    test_set = tf.data.Dataset.from_tensor_slices(images)
    test_set = test_set.map(load_image)  # 1)
    test_set = test_set.map(preprocess)
    test_set = test_set.map(extract_crop)
    test_set = test_set.batch(config.BATCH_SIZE, drop_remainder=True)
    return test_set, test_step
