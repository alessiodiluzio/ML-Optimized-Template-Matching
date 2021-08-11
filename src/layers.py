""" Implementation of net Layers"""

import tensorflow as tf


""" 
Implementation of a classical convolutional layer followed by Batch Normalization adapted to a Siamese network.
Since it is thought to be used in a Siamese network, Kernel Weights are shared between two different
input channel. 
(Each input channel has its own bias since the input size, height and width, can be different)
"""


class SiameseConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, activation, name, **kwargs):
        super(SiameseConv2D, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.activation = activation
        self.batch_normalization = None
        self.W = tf.constant(0.)
        self.b_source = tf.constant(0.)
        self.b_template = tf.constant(0.)

    def build(self, input_shape):
        """
        Build convolutional layer on its first call.
        :param input_shape: [shape of x, shape of z] where x and z are the feature maps
        extracted from source and template image at current stage.
        Shape is in the form [B, H, W, C] where B is the batch size, H, W and C are
        Height, Width and Channels of feature map.
        Weights of kernel (shared) and biases are built according to the shape of x and z.
        :return:
        """
        w_shape = self.kernel_size + (input_shape[0][-1], self.filters)
        b_source_shape = (int((input_shape[0][1] - self.kernel_size[0])/self.strides) + 1,
                          int((input_shape[0][2] - self.kernel_size[1])/self.strides) + 1,
                          self.filters)
        b_template_shape = (int((input_shape[1][1] - self.kernel_size[0]) / self.strides) + 1,
                            int((input_shape[1][2] - self.kernel_size[1]) / self.strides) + 1,
                            self.filters)
        self.W = self.add_weight(name='kernel', shape=w_shape, trainable=True,
                                 initializer=tf.keras.initializers.GlorotUniform)
        self.b_source = self.add_weight(name='bias_s', shape=b_source_shape, initializer='zeros', trainable=True)
        self.b_template = self.add_weight(name='bias_t', shape=b_template_shape, initializer='zeros', trainable=True)
        self.batch_normalization = tf.keras.layers.BatchNormalization(axis=-1)

    @tf.function
    def call(self, inputs, training=False, **kwargs):
        """
        Apply convolution to source and template images using same kernel but different biases (due to different size of
        source and template), then, if the layer is called during the training phase, batch normalization is applied.
        :param inputs: [x, z] where x and z are the feature maps
        extracted from source and template image at current stage.
        :param training: True if batch normalization is applied after convolution (training phase)
        false on the contrary (inference).
        :param kwargs:
        :return: [x, z] where x and z are the maps of feature extracted from the source and template images.
        """
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


"""
Implementation of the correlation filter described in https://arxiv.org/pdf/1606.09549.pdf
and implemented at https://github.com/torrvision/siamfc-tf/blob/master/src/siamese.py.
In a nutshell, Source image feature maps are convolved using template feature maps as kernel.
"""


class CorrelationFilter(tf.keras.layers.Layer):

    def __init__(self):
        super(CorrelationFilter, self).__init__(name='correlation_filter')
        self.perm = tf.convert_to_tensor([1, 2, 0, 3])
        self.stride = [1, 1, 1, 1]
        self.padding = 'VALID'
        self.batch_size = None
        self.source_output_dim = None
        self.template_output_dim = None
        self.output_channels = None
        self.batch_normalization = None

    def build(self, input_shape):
        """
         Build correlation filter on its first call, Shape of bias is chosen according to net output dim.
        :param input_shape: [shape of x, shape of z]  where x and z are the feature maps
        extracted from source and template image at current stage.
        Shape is in the form [B, H, W, C] where B is the batch size, H, W and C are
        Height, Width and Channels of feature map.
        :return:
        """
        self.source_output_dim = input_shape[0][1]
        self.template_output_dim = input_shape[1][1]
        self.batch_size = input_shape[0][0]
        self.output_channels = input_shape[0][3]
        self.batch_normalization = tf.keras.layers.BatchNormalization(axis=-1)

    @tf.function
    def call(self, inputs, training=False, **kwargs):
        """
        Implementation of correlation filter,
        refer to https://github.com/torrvision/siamfc-tf/blob/master/src/siamese.py and
        https://arxiv.org/pdf/1606.09549.pdf.
        :param inputs:  [x, z] where x and z are the feature maps
        extracted from source and template image at current stage.
        :param training: True if batch normalization is applied after convolution (training phase)
        false on the contrary (inference).
        :param kwargs: unused
        :return: Heatmap score for the localization of template in source image.
        """

        # z, x are [B, H, W, C]

        x = tf.transpose(inputs[0], perm=[1, 2, 0, 3])
        z = tf.transpose(inputs[1], perm=[1, 2, 0, 3])
        # z, x are [H, W, B, C]

        x = tf.reshape(x, (1, self.source_output_dim, self.source_output_dim,
                           self.batch_size * self.output_channels))
        # x is [1, Hx, Wx, B*C]

        z = tf.reshape(z, (self.template_output_dim, self.template_output_dim,
                           self.batch_size * self.output_channels, 1))
        # z is [Hz, Wz, B*C, 1]

        net_final = tf.nn.depthwise_conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID')
        # final is [1, Hf, Wf, BC]

        net_final = tf.split(net_final, self.batch_size, axis=3)
        net_final = tf.concat(net_final, axis=0)
        # final is [B, Hf, Wf, C]

        net_final = tf.expand_dims(tf.reduce_sum(net_final, axis=3), axis=3)
        # final is [B, Hf, Wf, 1]

        net_final = self.batch_normalization(net_final, training=training)

        return net_final
