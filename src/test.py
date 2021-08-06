import tensorflow as tf
import os

from src.utils import plot
from src.config import BATCH_SIZE
from src.dataset import get_test_set
from src.model import Siamese


def test(data_path):
    model = tf.keras.models.load_model('saved_model')
    # model_ckpt = tf.train.Checkpoint(net=model)
    # restored = model_ckpt.restore(tf.train.latest_checkpoint('checkpoint'))
    test_set, test_step = get_test_set(data_path)
    dest_path = 'image'

    for step, (image, template, _) in enumerate(test_set):
        prediction = model([image, template], training=False)
        for i in range(BATCH_SIZE):
            filename = f'test_prediction_{step}_{i}_model_{model.name}.jpg'
            file_path = os.path.join(dest_path, filename)
            plot(image[[i]], template[i], logit=prediction[i], target='save', dest=file_path)
