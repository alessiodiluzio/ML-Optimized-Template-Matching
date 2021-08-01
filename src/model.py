import tensorflow as tf
import os
from src.layers import SiameseConv2D, CorrelationFilter


class Siamese(tf.keras.Model):

    def __init__(self):
        super(Siamese, self).__init__(name='Siamese')
        self._alex_net_encoder = None
        self._correlation_filter = None
        self._up_sample = None

    def build(self, input_shape):
        self._alex_net_encoder = AlexnetEncoder()
        self._correlation_filter = CorrelationFilter()
        self._up_sample = tf.keras.layers.UpSampling2D(size=(15, 15))

    def call(self, input_tensor, training=False, **kwargs):
        x, z = self._alex_net_encoder(input_tensor, training)
        corr = self._correlation_filter([x, z])
        net_final = self._up_sample(corr)
        return net_final

    def save_model(self, checkpoint_dir):
        tf.saved_model.save(self._alex_net_encoder, os.path.join(checkpoint_dir, AlexnetEncoder.get_name()))
        tf.saved_model.save(self._correlation_filter, os.path.join(checkpoint_dir, CorrelationFilter.get_name()))

    def load_model(self, checkpoint_dir):
        self._alex_net_encoder = tf.keras.models.load_model(os.path.join(checkpoint_dir,
                                                                         AlexnetEncoder.get_name()))
        self._correlation_filter = tf.keras.models.load_model(os.path.join(checkpoint_dir,
                                                                           CorrelationFilter.get_name()))


class AlexnetEncoder(tf.keras.Model):

    def __init__(self):
        super(AlexnetEncoder, self).__init__(name='alexnet_encoder')
        self.conv1 = None
        self.pool1 = None

        self.conv2 = None
        self.pool2 = None

        self.conv3 = None
        self.conv4 = None
        self.conv5 = None

    def build(self, input_shape):
        self.conv1 = SiameseConv2D(filters=96, kernel_size=(11, 11), strides=2,
                                   padding='valid', activation=tf.keras.activations.relu, name='Conv_1')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, name='Max_Pool_1')

        self.conv2 = SiameseConv2D(filters=256, kernel_size=(5, 5), strides=1,
                                   padding='valid', activation=tf.keras.activations.relu, name='Conv_2')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, name='Max_Pool_2')

        self.conv3 = SiameseConv2D(filters=192, kernel_size=(3, 3), strides=1,
                                   padding='valid', activation=tf.keras.activations.relu, name='Conv_3')
        self.conv4 = SiameseConv2D(filters=192, kernel_size=(3, 3), strides=1,
                                   padding='valid', activation=tf.keras.activations.relu, name='Conv_4')
        self.conv5 = SiameseConv2D(filters=128, kernel_size=(3, 3), strides=1, padding='valid',
                                   activation=None, name='Conv_5')

    def call(self, input_tensor, training=False, **kwargs):
        output = self.conv1(input_tensor, training)

        x = self.pool1(output[0])
        z = self.pool1(output[1])

        x, z = self.conv2([x, z], training)

        x = self.pool2(x)
        z = self.pool2(z)

        x, z = self.conv3([x, z], training)
        x, z = self.conv4([x, z], training)
        x, z = self.conv5([x, z], training)
        return x, z

    @staticmethod
    def get_name():
        return 'alexnet_encoder'


