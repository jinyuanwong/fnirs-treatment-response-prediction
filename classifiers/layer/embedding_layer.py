from tensorflow.keras import layers
import tensorflow as tf

class EmbeddingLayer(layers.Layer):
    def __init__(self, d_model, filters, kernel_size, strides, l2_rate, name="EmbeddingLayer"):
        super(EmbeddingLayer, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride_size = strides
        self.d_model = d_model
        self.l2_rate = l2_rate
    # Why we must have a input_shape but we can not use it, or it will have issues.

    def build(self, input_shape):  # input_shape : batch, channel_dimension, sample_points, HbO/HbR(1,2)
        self.cnn_1 = layers.Conv2D(filters=self.filters,
                                   kernel_size=self.kernel_size,
                                   strides=self.stride_size)

        # self.size_1 = (input_shape[1] - self.kernel_size[0]) // self.stride_size[0] + 1
        # print(f'here size_1 = {self.size_1}')
        self.out_dimension = (
            input_shape[2] - self.kernel_size[1]) // self.stride_size[1] + 1  # {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1 |n=len, p=padding, f=kernel, s=stride ;

        # check_shape = (None, 52, x * y) # using the einsum can be more elegant
        # equal to layers.Reshape((-1, self.out_dimension * self.filters)) , batch_size is ignored
        self.flatten = layers.Reshape((-1, self.out_dimension * self.filters))
        self.lin = layers.Dense(
            self.d_model, kernel_regularizer=tf.keras.regularizers.l2(self.l2_rate))
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        outputs = self.cnn_1(inputs)

        # # check here updated at 14 July 2023 by adding the transpose operation if you can not have good result from this time,
        # outputs = tf.transpose(outputs, perm=[0, 2, 1, 3]) this step change the dimension of channel and sample point, which is not a good choice because you will get (None, 128, channel * output_channel_of_CNN), you lose the comparison of different channel.
        outputs = self.flatten(outputs)
        outputs = self.lin(outputs)
        outputs = self.norm(outputs)
        return outputs