import os
import tensorflow as tf

from matplotlib import pyplot as plt
from src import X_1, X_2, Y_1, Y_2, IMAGE_DIM, CROP_SIZE


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
    device = 'cpu'
    if len(tf.config.list_physical_devices('GPU')) > 0:# or OS == 'Darwin':
        device = 'gpu'
    return device


# def make_prediction(boxes, x_scale_factor=CROP_SIZE, y_scale_factor=CROP_SIZE, outer_box_dim=IMAGE_DIM):
#
#     x1 = boxes[X_1]
#     y1 = boxes[Y_1]
#
#     x2 = boxes[X_2]
#     y2 = boxes[Y_2]
#
#     # x1 = tf.cast(x1 * x_scale_factor + x_scale_factor / 2, dtype=tf.int32)
#     # y1 = tf.cast(y1 * y_scale_factor + y_scale_factor / 2, dtype=tf.int32)
#     # x2 = tf.cast(x2 * x_scale_factor + x_scale_factor / 2, dtype=tf.int32)
#     # y2 = tf.cast(y2 * y_scale_factor + y_scale_factor / 2, dtype=tf.int32)
#
#     x1 = tf.cast(x1, dtype=tf.int32)
#     x2 = tf.cast(x2, dtype=tf.int32)
#     y1 = tf.cast(y1, dtype=tf.int32)
#     y2 = tf.cast(y2, dtype=tf.int32)
#
#     # x1_tmp = tf.minimum(x1, x2)
#     # x2 = tf.maximum(x1, x2)
#     # x1 = x1_tmp
#
#     # y1_tmp = tf.maximum(y1, y2)
#     # y2 = tf.maximum(y1, y2)
#     # y1 = y1_tmp
#
#     x1 = tf.maximum(0, x1)
#     x1 = tf.minimum(outer_box_dim, x1)
#
#     y1 = tf.maximum(0, y1)
#     y1 = tf.minimum(outer_box_dim, y1)
#
#     x2 = tf.maximum(0, x2)
#     x2 = tf.minimum(outer_box_dim, x2)
#
#     y2 = tf.maximum(0, y2)
#     y2 = tf.minimum(outer_box_dim, y2)
#
#     x1, x2 = tf.cond(tf.greater(x1, x2), lambda: [0, 0], lambda: [x1, x2])
#     y1, y2 = tf.cond(tf.greater(y1, y2), lambda: [0, 0], lambda: [y1, y2])
#
#     boxes = tf.stack([x1, y1, x2, y2], axis=0)
#     prediction = make_label(boxes, outer_box_dim)
#     return prediction


def make_label(boxes, outer_box_dim=IMAGE_DIM):

    x1 = boxes[X_1]
    y1 = boxes[Y_1]
    x2 = boxes[X_2]
    y2 = boxes[Y_2]

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


def get_zero_base_label(labels):
    label_true = tf.cast((labels + 1) / 2, dtype=tf.int32)
    label_false = tf.cast((labels - 1) / 2, dtype=tf.int32)
    return label_true + label_false


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


def get_balance_factor():
    return tf.cast(tf.divide(tf.multiply(CROP_SIZE, CROP_SIZE), tf.multiply(IMAGE_DIM, IMAGE_DIM)), dtype=tf.float32)
