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
from utils.utils import save_logs
import tensorflow.keras as keras
from tensorflow.keras.initializers import HeNormal
from sklearn.metrics import auc, accuracy_score, roc_curve, recall_score
import tensorflow_addons as tfa
import random
import os
import pandas as pd
import math

from tensorflow.keras.callbacks import EarlyStopping


# Transformer was based on
#   Zenghui Wang' Pytorch implementation. https://github.com/wzhlearning/fNIRS-Transformer/blob/main/model.py
# Adapted to Tensorflow by Jinyuan Wang
# MIT License

# Wavelet Transform was based on
#   1. Alexander Neergaard Olesen's CWT Python implementation. https://github.com/neergaard/CWT
#   2. Geir Drange's Keras implementation. https://www.kaggle.com/code/mistag/wavelet1d-custom-keras-wavelet-transform-layer
# Adapted to Tensorflow by Jinyuan Wang
# MIT License


"""
put this function into the utils as well: 
but remember that do not modify utils so much.
"""
check_shape = [0]


def read_past_value(directory, check_metrice):
    hist_loc = directory + 'history.csv'
    if os.path.exists(hist_loc):
        history = pd.read_csv(hist_loc)
        return np.max(history['val_' + check_metrice])
    else:
        return 0


def read_current_value(Y_pred, Y_true, check_metrice):

    if check_metrice == 'accuracy':
        return accuracy_score(Y_true, np.argmax(Y_pred, axis=1))
    else:
        raise ('You have not create a calculation for: ' + check_metrice)


def check_if_save_model(output_directory, Y_pred, Y_true, check_metrice):
    past_metrice = read_past_value(output_directory, check_metrice)
    current_metrice = read_current_value(Y_pred, Y_true, check_metrice)
    print(f"Current accuracy: {current_metrice}")
    if current_metrice > past_metrice:
        return True
    return False


"""
This is used to calculate the scales of wavalet transform

inputs: 

    input_shape: (batch_size, time_sampling_point)
    nv: number of voices

Returns:

    scales: a list of number 

"""


def calculate_scales(input_shape, nv, sr, flow, fhigh):
    max_scale = input_shape[1] // (np.sqrt(2) * 2)
    max_scale = np.floor(nv * np.log2(max_scale))  # 89
    scales = 2 * (2**(1/nv)) ** np.arange(0, max_scale + 1)
    frequencies = sr * (6 / (2 * np.pi)) / scales
    frequencies = frequencies[frequencies >= flow]
    scales = scales[0:len(frequencies)]

    frequencies = frequencies[frequencies <= fhigh]
    scales = scales[len(scales)-len(frequencies):len(scales)]
    return scales


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # it is really import to cast into tf.float 32 to train the models
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}


def CreateWaveletTransform(input_shape, nv, sr, flow, fhigh, trainable):
    scales = calculate_scales(
        input_shape, nv, sr, flow, fhigh)
    # wavft

    padvalue = input_shape[1] // 2
    n = padvalue*2 + input_shape[1]
    omega = tf.range(1, math.floor(n/2) + 1,
                     dtype=tf.float64) * (2 * np.pi) / n
    minus_omega = -tf.reverse(omega, axis=[0])
    omega = tf.concat(
        [tf.constant([0.], dtype=tf.float64), omega, minus_omega], axis=0)

    gc, mul = 6, 2  # morl parameters: https://github.com/neergaard/CWT/blob/master/cwt.py
    _wft = mul * tf.exp(-(tf.expand_dims(scales, axis=1) * omega - gc)
                        ** 2 / 2.0) * tf.cast(omega > 0, dtype=tf.float64)

    wft = tf.Variable(
        _wft, trainable=trainable)

    wft_img = tf.Variable(
        _wft, trainable=trainable)

    return tf.cast(tf.complex(wft, wft_img), dtype=tf.complex64)


class Wavelet1D(keras.layers.Layer):
    def __init__(self, nv=12, sr=1., flow=0., fhigh=100, batch_size=None, trainable=True):
        super(Wavelet1D, self).__init__()
        assert fhigh > flow, 'fhigh parameters must be > flow'
        # assert batch_size != None, 'batch size must be set'

        # self.batch_size = batch_size
        self.nv = nv  # number of voices
        self.sr = sr  # sample rate
        self.flow = flow  # lowest frequency of interest
        self.fhigh = fhigh  # highest frequency of interest
        self.trainable = trainable

    def build(self, input_shape):
        assert len(input_shape) == 2, 'Input dimension must be 2!(batch, sampledata) Dimension is {}'.format(
            len(input_shape))

        # with tf.device('/GPU:0'):
        scales = calculate_scales(
            input_shape, self.nv, self.sr, self.flow, self.fhigh)
        self.padvalue = input_shape[1] // 2
        self.n = self.padvalue*2 + input_shape[1]
        self.num_scales = scales.shape[-1]
        self.wft = CreateWaveletTransform(
            input_shape, self.nv, self.sr, self.flow, self.fhigh, self.trainable)

    def call(self, inputs):
        # with tf.device('/GPU:0'):
        x = tf.concat((tf.reverse(inputs[:, 0:self.padvalue], axis=[
            1]), inputs, tf.reverse(inputs[:, -self.padvalue:], axis=[1])), axis=1)  # x - (None, 249) = 62 + 125 + 62
        f = tf.signal.fft(tf.cast(x, tf.complex64))  # (None, 249)
        fr = tf.repeat(f, self.num_scales, axis=0)  # (None, 249)

        # (None, 88, 249),  self.n = 249
        fr = tf.reshape(fr, (-1, self.num_scales, self.n))

        # (None, 88, 249), wft - (88, 249)
        # tf.cast(self.wft, tf.complex64)
        cwtcfs = tf.signal.ifft(fr * self.wft)
        cfs = cwtcfs[:, :, self.padvalue:self.padvalue +
                     inputs.shape[-1]]  # (None, 88, 125)
        return tf.math.abs(cfs)


class Scaler(keras.layers.Layer):
    def __init__(self, upper=1.0):
        super(Scaler, self).__init__()
        # upper value (typically 1.0 or 255.0 for image CNNs)
        self.upper = tf.cast(upper, dtype=tf.float32)

    def call(self, inputs):
        min_val = tf.math.reduce_min(inputs)
        max_val = tf.math.reduce_max(tf.math.subtract(inputs, min_val))
        return tf.math.multiply(tf.math.subtract(inputs, min_val), self.upper/max_val)


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


class WaveletLayer(keras.layers.Layer):
    def __init__(self, d_model, filter, kernel_size, stride_size, l2_rate, scaler_max, number_of_voices=16, flow=0, fhigh=100, name="EmbeddingLayer"):
        super(WaveletLayer, self).__init__(name=name)
        self.d_model = d_model
        self.filters = filter
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.l2_rate = l2_rate
        self.scaler_max = scaler_max
        self.nv = number_of_voices
        self.flow = flow
        self.fhigh = fhigh
        # self.batch_size = batch_size

    def build(self, input_shape):
        # inputs = input(shape=input_shape)
        self.channel_wavelet = [Wavelet1D(
            nv=self.nv, sr=1, flow=self.flow, fhigh=self.fhigh) for _ in range(input_shape[1])]

        # self.scaler = Scaler(upper=self.scaler_max)

        self.pool2d_1 = layers.AveragePooling2D(
            pool_size=(4, 4), strides=(4, 4))
        self.conv2d1 = layers.Conv2D(filters=128,
                                     kernel_size=3,
                                     strides=2)
        self.conv2d2 = layers.Conv2D(filters=128,
                                     kernel_size=3,
                                     strides=2)
        self.conv2d3 = layers.Conv2D(filters=52,
                                     kernel_size=3,
                                     strides=2)

        self.lin = tf.keras.layers.Dense(
            self.d_model, kernel_regularizer=tf.keras.regularizers.l2(self.l2_rate))
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):

        outputs = [self.channel_wavelet[i](
            inputs[:, i, :, 0]) for i in range(inputs.shape[1])]
        # outputs = tf.math.abs(outputs)

        outputs = tf.stack(outputs, axis=3)
        #
        # outputs = self.scaler(outputs)

        outputs = self.pool2d_1(outputs)
        outputs = self.conv2d1(outputs)
        outputs = self.conv2d2(outputs)
        outputs = self.conv2d3(outputs)
        # outputs = self.pool2d_2(outputs)
        # outputs = self.pool2d_3(outputs)
        # print(f' - outputs x1 - {outputs.shape}')

        # from (batch, frequency, time, channel) to (batch, channel, frequency, time)
        outputs = tf.transpose(outputs, (0, 3, 1, 2))

        # print(f' - outputs x2 - {outputs.shape}')

        # output_shape = outputs.shape
        # outputs = tf.keras.layers.Reshape((-1))(outputs)
        outputs = tf.reshape(outputs, (-1, tf.shape(outputs)
                             [1], tf.shape(outputs)[2] * tf.shape(outputs)[3]))  # （batch, frequency, time_sampling * channels
        # outputs = self.norm(outputs)
        # print(f' - outputs x3 - {outputs.shape}')
        # print(f' - outputs x - {outputs.shape}')
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


class ClsPositionEncodingLayer(layers.Layer):
    def __init__(self, input_channel, kenerl_size, strides, d_model, dropout_rate):
        super(ClsPositionEncodingLayer, self).__init__()

        patch = (input_channel - kenerl_size[0]) // strides[0] + 1
        self.cls_token_patch = tf.Variable(tf.random.normal((1, 1, d_model)))
        self.pos_embedding_patch = tf.Variable(
            tf.random.normal((1, patch+1, d_model)))
        self.dropout_patch = layers.Dropout(dropout_rate)

    def call(self, inputs):
        cls_token_patch_tiled = tf.tile(
            self.cls_token_patch, [tf.shape(inputs)[0], 1, 1])
        pos_embedding_path_tiled = tf.tile(self.pos_embedding_patch, [
            tf.shape(inputs)[0], 1, 1])

        outputs = tf.concat([cls_token_patch_tiled, inputs], axis=1)
        outputs = outputs + pos_embedding_path_tiled
        outputs = self.dropout_patch(outputs)
        return outputs


class Transformer(tf.keras.Model):
    # input_shape = (None, channel_size, sample_point, data - HbO))
    def __init__(self,
                 inputs_shape,
                 num_class,
                 dropout_rate,
                 d_model,
                 filters,
                 kernel_size,
                 strides,
                 n_layers,
                 FFN_units,
                 n_heads,
                 activation,
                 num_of_last_dense,
                 l2_rate,
                 scaler_max,
                 number_of_voices):
        super(Transformer, self).__init__()

        self.l2_rate = l2_rate
        self.FFN_units = FFN_units

        # self.embedding_1 = EmbeddingLayer(
        #     d_model, filters, kernel_size[0], strides[0], l2_rate, name='embedding_1')
        # self.embedding_2 = EmbeddingLayer(
        #     d_model, filters, kernel_size[1], strides[1], l2_rate, name='embedding_2')
        self.embedding_wavelet = WaveletLayer(
            d_model, filters, kernel_size[2], strides[2], l2_rate, scaler_max, number_of_voices, name="embedding_3")

        emb_3_size_0 = inputs_shape[1]  #
        # emb_3_size_0 = calculate_scales((None, inputs_shape[2]), number_of_voices, sr=1, flow=0, fhigh=100).shape[0]//4 # / 2 is because the pool size is (2,2)
        # emb_3_size_0 = (emb_3_size_0 - kernel_size[2][0]) // strides[2][0] + 1

        # self.cls_pos_encoding_1 = ClsPositionEncodingLayer(
        #     input_channel=inputs_shape[1], kenerl_size=kernel_size[0], strides=strides[0], d_model=d_model, dropout_rate=dropout_rate)
        # self.cls_pos_encoding_2 = ClsPositionEncodingLayer(
        #     input_channel=inputs_shape[1], kenerl_size=kernel_size[1], strides=strides[1], d_model=d_model, dropout_rate=dropout_rate)
        # self.cls_pos_encoding_3 = ClsPositionEncodingLayer(input_channel=emb_3_size_0, kenerl_size=[
        #                                                    1, 1], strides=[1, 1], d_model=d_model, dropout_rate=dropout_rate)

        self.cls_pos_encoding_3 = ClsPositionEncodingLayer(input_channel=emb_3_size_0, kenerl_size=[
                                                           1, 1], strides=[1, 1], d_model=d_model, dropout_rate=dropout_rate)

        # self.encoder_1 = Encoder(n_layers,
        #                          FFN_units,
        #                          n_heads,
        #                          dropout_rate,
        #                          activation,
        #                          name="encoder_1")

        # self.encoder_2 = Encoder(n_layers,
        #                          FFN_units,
        #                          n_heads,
        #                          dropout_rate,
        #                          activation,
        #                          name="encoder_2")

        self.encoder_3 = Encoder(n_layers,
                                 FFN_units,
                                 n_heads,
                                 dropout_rate,
                                 activation,
                                 name="encoder_3")

        # self.global_average_pooling_1 = layers.GlobalAveragePooling1D(
        #     data_format='channels_first', keepdims=False)
        # self.global_average_pooling_2 = layers.GlobalAveragePooling1D(
        #     data_format='channels_first', keepdims=False)
        self.global_average_pooling_3 = layers.GlobalAveragePooling1D(
            data_format='channels_first', keepdims=False)

        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):

        # emb_output_1 = self.embedding_1(inputs)
        # emb_output_2 = self.embedding_2(inputs)
        emb_output_3 = self.embedding_wavelet(inputs)

        # output_1 = self.cls_pos_encoding_1(emb_output_1)
        # output_2 = self.cls_pos_encoding_2(emb_output_2)
        output_3 = self.cls_pos_encoding_3(emb_output_3)

        # output_1 = self.encoder_1(output_1)
        # output_2 = self.encoder_2(output_2)
        output_3 = self.encoder_3(output_3)

        # output_1 = self.global_average_pooling_1(output_1)
        # output_2 = self.global_average_pooling_2(output_2)
        output_3 = self.global_average_pooling_3(output_3)

        # output = tf.concat([output_1, output_2, output_3], axis=1)  #

        output = self.norm(output_3)

        return output


"""

Input 

"""


class Classifier_Transformer():
    def __init__(self, output_directory, callbacks, input_shape, epochs):
        # input_shape = (200, 52, 128, 1)

        self.output_directory = output_directory
        self.callbacks = callbacks
        self.epochs = epochs

        # 随机给定超参数进行训练
        # 32#random.choice([16, 32, 48])  # 128 256
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=100)
        self.callbacks.append(early_stopping)
        # 32  # random.choice([128]), if you can not load the model, this parameter is not the reason
        self.batch_size = 32  # random.choice([32, 128])
        kernel_size_1 = (4, 5)  # 2, 3, 4
        stride_size_1 = (1, 2)
        kernel_size_2 = (1, 5)  # 2: random.randint(2,8)  (2,5 are the best)
        stride_size_2 = (1, 2)
        kernel_size_3 = (4, 6)
        stride_size_3 = (2, 2)
        kernel_size = [kernel_size_1, kernel_size_2, kernel_size_3]
        stride_size = [stride_size_1, stride_size_2, stride_size_3]
        # random.choice([2, 3, 4, 5, 6, 7, 8]) 6,7 are the best
        # random.choice([4, 24])  # random.choice([12, 24, 36])
        output_channel = 4  # random.choice([4, 8, 12, 16, 20, 24])  # 24
        # random.choice([64, 256])# 64 #
        d_model = 64  # random.choice([64, 128, 256])
        dropout_rate = 0.4
        # random.choice([4, 12])  # random.randint(10, 12)
        n_layers = 12  # random.choice([12, 8, 16])
        FFN_units = 256  # random.choice([64, 128, 256, 512])  # 512, 64, 128,
        n_heads = 4  # random.choice([4, 8])  # 2
        #   # random.choice(['relu', 'gelu'])
        activation = 'gelu'  # random.choice(['relu', 'gelu'])
        # warmup_step random.choice([100,200,300,400,500,1000,2000])
        warmup_step = 200  # random.choice([200, 400, 600, 800, 1000])
        # random.choice([0.98, 0.99, 0.999])
        adam_beta_1, adam_beta_2 = 0.9, 0.999
        num_of_last_dense = 2  # random.randint(0, 3)
        l2_rate = 0.001  # random.choice([0.001,0.002,0.003,0.004,0.005])
        scaler_max = 1  # random.choice([1, 5, 255])
        number_of_voices = 16  # random.choice([16, 24, 32])
        num_class = 2
        learning_rate = CustomSchedule(
            d_model * FFN_units * n_layers, warmup_step)
        optimizer = tf.keras.optimizers.AdamW(learning_rate,
                                              beta_1=adam_beta_1,
                                              beta_2=adam_beta_2,
                                              epsilon=1e-9)

        inputs = tf.keras.Input(shape=input_shape[1:])

        outputs = Transformer(input_shape,
                              num_class,
                              dropout_rate,
                              d_model,
                              output_channel,
                              kernel_size,
                              stride_size,
                              n_layers,
                              FFN_units,
                              n_heads,
                              activation,
                              num_of_last_dense,
                              l2_rate,
                              scaler_max,
                              number_of_voices)(inputs)

        for i in range(num_of_last_dense):
            outputs = layers.Dense(FFN_units/(2**i),
                                   activation=activation,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(outputs)
        outputs = layers.Dense(num_class, activation='softmax')(outputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.summary()
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

        self.hyperparameters = {
            "only wavelet layer": True,
            "batch_size": self.batch_size,
            "kernel_size_1": kernel_size_1,
            "stride_size_1": stride_size_1,
            "kernel_size_2": kernel_size_2,
            "stride_size_2": stride_size_2,
            "kernel_size_3": kernel_size_3,
            "stride_size_3": stride_size_3,
            "output_channel": output_channel,
            "d_model": d_model,
            "dropout_rate": dropout_rate,
            "n_layers": n_layers,
            "FFN_units": FFN_units,
            "n_heads": n_heads,
            "num_class": num_class,
            "activation": activation,  # "lr": lr,
            "num_of_last_dense": num_of_last_dense,
            "l2_rate": l2_rate,
            "scaler_max": scaler_max,
            "number_of_voices": number_of_voices,
            "learning_rate": f"CustomSchedule(d_model * FFN_units * n_layers, {warmup_step})",
            "optimizer": f"tf.keras.optimizers.AdamW(learning_rate, beta_1={adam_beta_1}, beta_2={adam_beta_2}, epsilon=1e-9)"  # "optimizer = tf.keras.optimizers.RMSprop(\
            # learning_rate={lr}, rho=0.9, momentum={momentum}, epsilon=1e-07, centered=False)"#
        }
        print(f'hyperparameters: {self.hyperparameters}')

    def fit(self, data, label, X_test, Y_test):
        start_time = time.time()
        hist = self.model.fit(
            x=data,
            y=label,
            validation_data=(X_test, Y_test),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=True,
            shuffle=True  # Set shuffle to True
        )

        self.model.load_weights(
            self.output_directory + 'checkpoint')
        Y_pred = self.model.predict(X_test)
        Y_pred_max = np.argmax(Y_pred, axis=1)
        Y_true = np.argmax(Y_test, axis=1)

        duration = time.time() - start_time

        if check_if_save_model(self.output_directory, Y_pred, Y_true, 'accuracy'):
            # save learning rate as well
            # Can ignore the result name which has beend set as None
            save_logs(self.model, self.output_directory, None,
                      hist, Y_pred_max, Y_true, duration,
                      lr=True,
                      is_saving_checkpoint=True,
                      hyperparameters=self.hyperparameters,
                      y_true_onehot=Y_test,
                      y_pred_onehot=Y_pred
                      )

    def predict(self):
        pass


# model = Transformer('transformer', None, None, (5, 52, 128, 1), 1)
