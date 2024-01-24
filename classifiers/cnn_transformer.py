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


class PositionalEncoding(layers.Layer):

    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_angles(self, pos, i, d_model):  # pos: (seq_length, 1) i: (1, d_model)
        # 2*(i//2) => if i = 5 -> ans = 4
        angles = 1 / np.power(10000., (2*(i//2)) / np.float32(d_model))
        return pos * angles  # (seq_length, d_model)

    def call(self, inputs):
        # input shape batch_size, seq_length, d_model
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        # Calculate the angles given the input
        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                 np.arange(d_model)[np.newaxis, :],
                                 d_model)
        # Calculate the positional encodings
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        # Expand the encodings with a new dimension
        pos_encoding = angles[np.newaxis, ...]

        return inputs + tf.cast(pos_encoding, tf.float32)


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


class ClsPositionEncodingLayer(layers.Layer):
    def __init__(self, input_channel, kenerl_size, strides, d_model, dropout_rate, name="ClsPositionEncodingLayer"):
        super(ClsPositionEncodingLayer, self).__init__(name=name)

        patch = (input_channel - kenerl_size[0]) // strides[0] + 1
        self.cls_token_patch = tf.Variable(tf.random.normal((1, 1, d_model)))
        self.pos_embedding = PositionalEncoding()
        self.dropout_patch = layers.Dropout(dropout_rate)

    def call(self, inputs):
        cls_token_patch_tiled = tf.tile(
            self.cls_token_patch, [tf.shape(inputs)[0], 1, 1])

        outputs = tf.concat([cls_token_patch_tiled, inputs], axis=1)
        outputs = self.pos_embedding(outputs)
        outputs = self.dropout_patch(outputs)
        return outputs


"""Previous CLS_Position encoding

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
        
"""


class Transformer(tf.keras.Model):
    # input_shape = (None, channel_size, sample_point, datapoint)
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
                 l2_rate):
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


class Classifier_Transformer():
    def __init__(self, output_directory, callbacks, input_shape, epochs, sweep_config, info):
        # input_shape = (200, 52, 128, 1)

        self.output_directory = output_directory
        self.callbacks = callbacks
        self.epochs = epochs

        # 随机给定超参数进行训练
        # 32#random.choice([16, 32, 48])  # 128 256
        early_stopping = EarlyStopping(monitor='val_loss', patience=100)
        self.info = info
        self.callbacks.append(early_stopping)
        # 32  # random.choice([128]) # 没有影响，不改变模型的结构 # 8 is very bad ~70%
        self.batch_size = 128
        kernel_size_1 = (4, 5)  # 2, 3, 4
        stride_size_1 = (1, 2)
        kernel_size_2 = (1, 5)  # 2: random.randint(2,8)  (2,5 are the best)
        stride_size_2 = (1, 2)
        kernel_size = [kernel_size_1, kernel_size_2]
        stride_size = [stride_size_1, stride_size_2]
        # random.choice([2, 3, 4, 5, 6, 7, 8]) 6,7 are the best
        # random.choice([4, 24])  # random.choice([12, 24, 36])
        output_channel = 4  # random.choice([3, 8, 24]) # 24
        # random.choice([64, 256])# 64 #
        d_model = 64  # 125# # random.choice([64, 128, 256])
        dropout_rate = 0.4
        # random.choice([4, 12])  # random.randint(10, 12)
        n_layers = 12  # random.choice([12, 8, 16])
        FFN_units = 256  # random.choice([64, 128, 256, 512])  # 512, 64, 128,
        n_heads = 4  # 5  # random.choice([4, 8])  # 2
        #   # random.choice(['relu', 'gelu'])
        activation = 'gelu'  # random.choice(['relu', 'gelu'])
        # warmup_step random.choice([100,200,300,400,500,1000,2000])
        warmup_step = 200
        # random.choice([0.98, 0.99, 0.999])
        adam_beta_1, adam_beta_2 = 0.9, 0.999
        num_of_last_dense = 2  # random.randint(0, 3)
        l2_rate = 0.001
        num_class = 2  # 2
        learning_rate = CustomSchedule(
            d_model * FFN_units * n_layers, warmup_step)
        optimizer = tf.keras.optimizers.AdamW(learning_rate,
                                              beta_1=adam_beta_1,
                                              beta_2=adam_beta_2,
                                              epsilon=1e-9)

        # If you change these two hyperparameters, remember to change the  self.hyperparameters

        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        inputs = tf.keras.Input(shape=input_shape[1:])

        output_1 = EmbeddingLayer(
            d_model, output_channel, kernel_size[0], stride_size[0], l2_rate, name='cnn_embedding_1')(inputs)
        output_2 = EmbeddingLayer(
            d_model, output_channel, kernel_size[1], stride_size[1], l2_rate, name='cnn_embedding_2')(inputs)

        output_1 = ClsPositionEncodingLayer(
            input_channel=input_shape[1], kenerl_size=kernel_size[0], strides=stride_size[0], d_model=d_model, dropout_rate=dropout_rate, name='CLS_pos_encoding_1')(output_1)
        output_2 = ClsPositionEncodingLayer(
            input_channel=input_shape[1], kenerl_size=kernel_size[1], strides=stride_size[1], d_model=d_model, dropout_rate=dropout_rate, name='CLS_pos_encoding_2')(output_2)

        output_1 = Transformer(input_shape,
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
                               l2_rate)(output_1)
        output_2 = Transformer(input_shape,
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
                               l2_rate)(output_2)

        outputs = tf.concat([output_1, output_2], axis=1)  #

        outputs = layers.LayerNormalization(epsilon=1e-6)(outputs)

        "Doing this in here is to get the layer[-2] feature"
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
            "Test": 'Adding CLS and using Traditional Position encoding (10000)and CNN encoding',
            "batch_size": self.batch_size,
            "kernel_size_1": kernel_size_1,
            "stride_size_1": stride_size_1,
            "kernel_size_2": kernel_size_2,
            "stride_size_2": stride_size_2,
            "kernel_size": kernel_size,
            "stride_size": stride_size,
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
            "learning_rate": f"CustomSchedule(d_model * FFN_units * n_layers, {warmup_step})",
            "optimizer": f"tf.keras.optimizers.AdamW(learning_rate, beta_1={adam_beta_1}, beta_2={adam_beta_2}, epsilon=1e-9)"  # "optimizer = tf.keras.optimizers.RMSprop(\
            # learning_rate={lr}, rho=0.9, momentum={momentum}, epsilon=1e-07, centered=False)"#
        }
        print(f'hyperparameters: {self.hyperparameters}')

    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        start_time = time.time()
        hist = self.model.fit(
            x=X_train,
            y=Y_train,
            validation_data=(X_val, Y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=True,
            shuffle=True  # Set shuffle to True
        )

        self.model.load_weights(
            self.output_directory + 'checkpoint')
        Y_pred = self.model.predict(X_test)
        Y_pred = np.argmax(Y_pred, axis=1)
        Y_true = np.argmax(Y_test, axis=1)

        duration = time.time() - start_time
        save_validation_acc(self.output_directory, np.argmax(self.model.predict(X_val), axis=1), np.argmax(Y_val, axis=1), self.info['monitor_metric'], self.info)
        if check_if_save_model(self.output_directory, Y_pred, Y_true, self.info['monitor_metric'], self.info):
            # save learning rate as well
            # Can ignore the result name which has beend set as None
            save_logs(self.model, self.output_directory, None,
                      hist, Y_pred, Y_true, duration,
                      lr=True,
                      is_saving_checkpoint=True,
                      hyperparameters=self.hyperparameters,
                      y_true_onehot=Y_test,
                      y_pred_onehot=tf.one_hot(Y_pred, depth=2).numpy()
                      )

        print(f'Training time is {duration}')

    def predict(self):
        pass
# model = Transformer('transformer', None, None, (5, 52, 128, 1), 1)
