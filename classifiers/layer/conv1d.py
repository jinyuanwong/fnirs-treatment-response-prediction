import tensorflow as tf
from tensorflow.keras import layers

from einops import rearrange

class conv1d_layer(layers.Layer):
    def __init__(self, args):
        super(conv1d_layer, self).__init__()
        self.args = args
        self.conv1d = layers.Conv1D(filters=args.model_input_dims, 
                                    kernel_size=args.conv_kernel_size,
                                    activation=args.activation,
                                    data_format='channels_first',)
    def call(self, inputs):
        x = rearrange(inputs, 'b d l -> b l d')
        # print('1 x shape:', x.shape)
        x = self.conv1d(x)
        # print('2 x shape:', x.shape)
        x = rearrange(x, 'b l d -> b d l')
        # print('3 x shape:', x.shape)
        return x