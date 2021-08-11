import tensorflow as tf
import numpy as np


def draw_bounding_box_from_heatmap(heatmap, bb_size):
    """
    Draw a Bounding Box proposal from an heatmap generated by Siamese FC Model
    :param heatmap: [H, W, 1] Tensor representing a map of scores
    :param bb_size: size of bounding box
    :return: [H, W, 1] heatmap with a Bounding Box drawn on top
    """
    argmax = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    y_center = argmax[0]
    x_center = argmax[1]
    heatmap = tf.expand_dims(heatmap, axis=0)
    window = bb_size/2.
    y_min = y_center - window
    x_min = x_center - window
    y_max = y_center + window
    x_max = x_center + window
    boxes = [y_min, x_min, y_max, x_max]
    boxes = tf.reshape(boxes, [1, 1, 4])
    boxes = tf.cast(boxes, dtype=tf.float32) / heatmap.shape[1]
    colors = np.array([[1.0, 1.0, 1.0]])
    return tf.image.draw_bounding_boxes(heatmap, boxes, colors)[0]
