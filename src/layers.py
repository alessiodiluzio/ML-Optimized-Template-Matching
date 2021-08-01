import tensorflow as tf
from src import BATCH_SIZE, IMAGE_OUTPUT_DIM, CROP_OUTPUT_DIM, OUTPUT_CHANNELS


class SiameseConv2D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides, padding, activation, name, **kwargs):
        super(SiameseConv2D, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.activation = activation
        self.batch_normalization = None
        self.W = tf.Variable(0.)
        self.b_source = tf.Variable(0.)
        self.b_template = tf.Variable(0.)

    def build(self, input_shape):
        w_shape = self.kernel_size + (input_shape[0][-1], self.filters)
        b_source_shape = (int((input_shape[0][1] - self.kernel_size[0])/self.strides) + 1,
                          int((input_shape[0][2] - self.kernel_size[1])/self.strides) + 1,
                          self.filters)
        b_template_shape = (int((input_shape[1][1] - self.kernel_size[0]) / self.strides) + 1,
                            int((input_shape[1][2] - self.kernel_size[1]) / self.strides) + 1,
                            self.filters)
        self.W = self.add_weight(name='kernel', shape=w_shape, trainable=True, initializer=tf.keras.initializers.GlorotUniform)
        self.b_source = self.add_weight(name='bias_s', shape=b_source_shape, initializer='zeros', trainable=True)
        self.b_template = self.add_weight(name='bias_t', shape=b_template_shape, initializer='zeros', trainable=True)
        self.batch_normalization = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs, training=False, **kwargs):
        x = inputs[0]
        z = inputs[1]
        x = tf.nn.conv2d(x, filters=self.W, strides=[1, self.strides, self.strides, 1], padding=self.padding) \
            + self.b_source
        z = tf.nn.conv2d(z, filters=self.W, strides=[1, self.strides, self.strides, 1], padding=self.padding) \
            + self.b_template
        x = self.batch_normalization(x, training=training)
        z = self.batch_normalization(z, training=training)
        if self.activation is not None:
            x = self.activation(x)
            z = self.activation(z)
        return x, z


# class Conv2DShared(tf.keras.layers.Layer):
#
#     def __init__(self, shared_layer, filters, kernel_size, strides, padding, activation, name, **kwargs):
#         super(Conv2DShared, self).__init__(name=name, **kwargs)
#         self.twin_layer = shared_layer
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.padding = padding.upper()
#         self.activation = activation
#         self.batch_normalization = tf.keras.layers.BatchNormalization(axis=-1)
#         self.b = 0
#
#     def build(self, input_shape):
#         b_shape = (int((input_shape[1] - self.kernel_size[0]) / self.strides) + 1,
#                    int((input_shape[2] - self.kernel_size[1]) / self.strides) + 1,
#                    self.filters)
#         self.b = self.add_weight(name='bias', shape=b_shape, initializer='zeros', trainable=True)
#
#     def call(self, inputs, training=False, **kwargs):
#         x = tf.nn.conv2d(inputs, filters=self.twin_layer.W, strides=[1, self.strides, self.strides, 1],
#                          padding=self.padding) + self.b
#         x = self.batch_normalization(x, training=training)
#         if self.activation is not None:
#             x = self.activation(x)
#         return x


class CorrelationFilter(tf.keras.layers.Layer):

    def __init__(self):
        super(CorrelationFilter, self).__init__(name='correlation_filter')
        self.b = tf.Variable(0.)
        self.perm = tf.convert_to_tensor([1, 2, 0, 3])
        self.stride = [1, 1, 1, 1]
        self.padding = 'VALID'

    def build(self, input_shape):
        b_shape = (1, 17, 17, BATCH_SIZE * OUTPUT_CHANNELS)

        self.b = self.add_weight(name='bias', shape=b_shape, initializer='zeros', trainable=True)

    def __call__(self, inputs, *args, **kwargs):

        # adattamento di _match_templates dal paper
        # di Luca Bertinetto https://github.com/torrvision/siamfc-tf/blob/master/src/siamese.py

        # z, x are [B, H, W, C]

        x = tf.transpose(inputs[0], perm=self.perm)
        z = tf.transpose(inputs[1], perm=self.perm)
        # z, x are [H, W, B, C]

        x = tf.reshape(x, (1, IMAGE_OUTPUT_DIM, IMAGE_OUTPUT_DIM, BATCH_SIZE * OUTPUT_CHANNELS))
        # x is [1, Hx, Wx, B*C]

        z = tf.reshape(z, (CROP_OUTPUT_DIM, CROP_OUTPUT_DIM, BATCH_SIZE * OUTPUT_CHANNELS, 1))
        # z is [Hz, Wz, B*C, 1]

        net_final = tf.nn.depthwise_conv2d(x, z, strides=self.stride, padding=self.padding) + self.b
        # final is [1, Hf, Wf, BC]

        net_final = tf.split(net_final, BATCH_SIZE, axis=3)
        net_final = tf.concat(net_final, axis=0)
        # final is [B, Hf, Wf, C]

        net_final = tf.expand_dims(tf.reduce_sum(net_final, axis=3), axis=3)
        # final is [B, Hf, Wf, 1]

        return net_final

    @staticmethod
    def get_name():
        return 'correlation_filter'







