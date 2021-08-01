import tensorflow as tf
import os

from src.model import Siamese
from src.utils import get_device


def test(test_set, output_path):

    device = get_device()
    siam_model = Siamese(checkpoint_dir="checkpoint", device=device)
    siam_model.load_model()
    # for b, (image, template, label) in enumerate(test_set):
        # prediction = tf.nn.sigmoid(siam_model.forward([image, template]))
        # for im in range(predictions[0]):
        #     save_plot(image, template, dest=os.path.join(output_path, str(im)+'.jpg'))
