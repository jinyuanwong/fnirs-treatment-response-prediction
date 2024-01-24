import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time

import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.optimizers as ko
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils.utils_mine import *
from utils.utils import *
import tensorflow.keras as keras
from tensorflow.keras.initializers import HeNormal
from sklearn.metrics import auc, accuracy_score, roc_curve, recall_score
import tensorflow_addons as tfa
import random
import os
import pandas as pd
import math
from tensorflow.keras.metrics import Recall
from tensorflow.keras.callbacks import EarlyStopping

# Transformer was based on
#   Zenghui Wang' Pytorch implementation. https://github.com/wzhlearning/fNIRS-Transformer/blob/main/model.py
# Adapted to Tensorflow by Jinyuan Wang
# MIT License


# import pywt
"""
put this function into the utils as well: 
but remember that do not modify utils so much.
"""

def scaled_fot_product_attention(queries, keys, values):

    product = tf.matmul(queries, keys, transpose_b=True)
    key_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_product = product / tf.math.sqrt(key_dim)

    attention = tf.matmul(tf.nn.softmax(scaled_product, axis=-1), values)
    return attention


class MultiHeadAttention(layers.Layer):

    def __init__(self, n_heads, name='multi_head_attention'):
        super(MultiHeadAttention, self).__init__(name=name)
        self.n_heads = n_heads

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads

        self.query_lin = layers.Dense(units=self.d_model)
        self.key_lin = layers.Dense(units=self.d_model)
        self.value_lin = layers.Dense(units=self.d_model)

        self.final_lin = layers.Dense(units=self.d_model)

    def split_proj(self, inputs, batch_size):  # inputs: (batch_size, seq_length, d_model)
        shape = (batch_size,
                 -1,
                 self.n_heads,
                 self.d_head)

        # outputs: (batch_size, seq_length, nb_proj, d_proj)
        splited_inputs = tf.reshape(inputs, shape=shape)
        # outputs: (batch_size, nb_proj, seq_length,  d_proj)
        return tf.transpose(splited_inputs, perm=[0, 2, 1, 3])

    def call(self, queries, keys, values):

        batch_size = tf.shape(queries)[0]
        queries = self.query_lin(queries)
        keys = self.key_lin(keys)
        values = self.value_lin(values)

        queries = self.split_proj(queries, batch_size)
        keys = self.split_proj(keys, batch_size)
        values = self.split_proj(values, batch_size)

        attention = scaled_fot_product_attention(queries, keys, values)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention,
                                      shape=(batch_size, -1, self.d_model))
        outputs = self.final_lin(concat_attention)
        return outputs


class EncoderLayer(layers.Layer):

    def __init__(self, FFN_units, n_heads, dropout_rate, activation, name='encoder_layer'):
        super(EncoderLayer, self).__init__(name=name)

        self.FFN_units = FFN_units
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.activation = activation

    def build(self, input_shape):
        self.d_model = input_shape[-1]
        self.multi_head_attention = MultiHeadAttention(self.n_heads)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn1_relu_gelu = layers.Dense(
            units=self.FFN_units, activation=self.activation)
        self.ffn2 = layers.Dense(units=self.d_model)
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attention = self.multi_head_attention(inputs, inputs, inputs)
        attention = self.dropout_1(attention)
        attention = self.norm_1(attention+inputs)

        outputs = self.ffn1_relu_gelu(attention)
        outputs = self.ffn2(outputs)
        outputs = self.dropout_2(outputs)

        outputs = self.norm_2(outputs + attention)
        return outputs


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


class Encoder(layers.Layer):
    def __init__(self,
                 n_layers,
                 FFN_units,
                 n_heads,
                 dropout_rate,
                 activation,
                 name="encoder"):
        super(Encoder, self).__init__(name=name)
        self.n_layers = n_layers
        self.enc_layers = [EncoderLayer(
            FFN_units, n_heads, dropout_rate, activation) for _ in range(n_layers)]

    def call(self, inputs):
        outputs = inputs
        for i in range(self.n_layers):
            outputs = self.enc_layers[i](outputs)
        return outputs


class Transformer(tf.keras.Model):
    # input_shape = (None, channel_size, sample_point, datapoint)
    def __init__(self,
                 dropout_rate,
                 n_layers,
                 FFN_units,
                 n_heads,
                 activation):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_layers,
                               FFN_units,
                               n_heads,
                               dropout_rate,
                               activation,
                               name="encoder_1")

        self.global_average_pooling = layers.GlobalAveragePooling1D(
            data_format='channels_first', keepdims=False)

        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):

        output_1 = self.encoder(inputs)  # self.encoder_1(output_1)

        output_1 = self.global_average_pooling(output_1)

        return output_1
