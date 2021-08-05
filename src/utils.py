import os
import tensorflow as tf

from matplotlib import pyplot as plt
from src import config


def get_filenames(path):
    out = []
    if os.path.exists(path):
        abs_path = os.path.abspath(path)
        names = os.listdir(path)
        names.sort()
        names = map(lambda name: os.path.join(abs_path, name), names)
        out = list(names)
    if len(out) == 0:
        print("Error Data Path is empty.")
        exit(1)
    return out


def get_device():
    device = '/CPU:0'
    if len(tf.config.list_physical_devices('GPU')) > 0 or config.OS == 'Darwin':
        device = '/GPU:0'
    return device


def make_label(boxes, outer_box_dim=config.IMAGE_DIM):

    x1 = boxes[config.X_1]
    y1 = boxes[config.Y_1]
    x2 = boxes[config.X_2]
    y2 = boxes[config.Y_2]

    x, y = x2 - x1, y2 - y1

    inner_box = tf.ones((x, y))
    paddings = [[y1, outer_box_dim - y2], [x1, outer_box_dim - x2]]
    ret = tf.pad(inner_box, paddings, mode='CONSTANT', constant_values=-1)
    ret = tf.cast(ret, dtype=tf.float32)
    ret = tf.zeros((outer_box_dim, outer_box_dim)) + ret
    ret = tf.expand_dims(ret, axis=-1)

    return ret


def create_label_mask(label_mask):
    label_mask = tf.argmax(label_mask, axis=-1)
    label_mask = label_mask[..., tf.newaxis]
    return label_mask


def plot(image, template, label=None, logit=None, target='save', dest='.'):
    n_plot = 2
    if label is not None:
        n_plot += 1
    if logit is not None:
        n_plot += 1
    fig = plt.figure()
    sub_plt = fig.add_subplot(1, n_plot, 1)
    sub_plt.set_title("Source")
    plt.imshow(image)
    sub_plt = fig.add_subplot(1, n_plot, 2)
    sub_plt.set_title("Template")
    plt.imshow(template)
    if label is not None:
        label = tf.squeeze(label, axis=-1)
        sub_plt = fig.add_subplot(1, n_plot, 3)
        sub_plt.set_title("Ground Truth")
        plt.imshow(label)
    if logit is not None:
        logit = tf.squeeze(logit, axis=-1)
        sub_plt = fig.add_subplot(1, n_plot, 4)
        sub_plt.set_title("Prediction")
        plt.imshow(logit)
    if target == 'save':
        plt.savefig(dest)
        plt.close()
    elif target == 'show':
        plt.show()


@tf.function
def get_zero_base_label(labels):
    return tf.divide(tf.add(labels, 1), 2, dtype=tf.int32)


def plot_dataset(dataset, samples, target='save', dest='.'):
    i = 0
    for image, template, label in dataset.take(samples):
        plot(image[i], template[i], label[i], target=target, dest=os.path.join(dest, str(i)+'.jpg'))
        i += 1


def plot_metrics(model_history, save_path):
    metric_names = [key.split('_')[1] for key in model_history if 'train' in key]
    for metric_name in metric_names:
        mv = model_history['val_'+metric_name]
        mt = model_history['train_'+metric_name]
        label_t = 'Training'
        label_v = 'Validation'
        plt.figure()
        color_v = 'red'
        color_t = 'blue'
        plt.plot(range(len(mt)), mt, color=color_t, linestyle='-', label=label_t + ' ' + metric_name)
        plt.plot(range(len(mv)), mv, color=color_v, linestyle='-', label=label_v + ' ' + metric_name)
        plt.title(metric_name)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name + ' value')
        if max(mt) < 1 and max(mv) < 1:
            plt.ylim([0, 1])
        plt.legend()
        plt.savefig(os.path.join(save_path, metric_name + '_' + str(len(mv)) + '.jpg'))
        plt.close()


def get_loss_balance_factor():
    return tf.cast(tf.divide(tf.multiply(config.CROP_SIZE, config.CROP_SIZE), tf.multiply(config.IMAGE_DIM, config.IMAGE_DIM)), dtype=tf.float32)
