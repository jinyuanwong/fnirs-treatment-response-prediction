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

from tensorflow.keras.callbacks import EarlyStopping

from transformers import TFBertModel
# import pywt
"""
put this function into the utils as well: 
but remember that do not modify utils so much.
"""
check_shape = [0]


def read_past_value(directory, check_metrice):
    print(f'directory : {directory}')

    # find if the history.file exists.
    if os.path.exists(directory):
        files = os.listdir(directory)
        for file in files:
            if file[:3] == 'his':
                location = directory + file
                history = pd.read_csv(location)
                return np.max(history['val_' + check_metrice])
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
        self.out_dimension = (
            input_shape[-2] - self.kernel_size[-1]) // self.stride_size[-1] + 1  # {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1 |n=len, p=padding, f=kernel, s=stride
        self.out_channels_size = (
            input_shape[1] - self.kernel_size[0]) // self.stride_size[0] + 1  # {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1 |n=len, p=padding, f=kernel, s=stride

        # check_shape = (None, 52, x * y)
        self.reshape_1 = layers.Reshape(
            (-1, self.out_dimension * self.filters, ))
        self.lin = layers.Dense(
            self.d_model, kernel_regularizer=tf.keras.regularizers.l2(self.l2_rate))
        self.reshape_2 = layers.Reshape(
            (-1, self.out_channels_size * self.d_model))

        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        outputs = self.cnn_1(inputs)
        # # check here updated at 14 July 2023 by adding the transpose operation if you can not have good result from this time,
        # outputs = tf.transpose(outputs, perm=[0, 2, 1, 3]) this step change the dimension of channel and sample point, which is not a good choice because you will get (None, 128, channel * output_channel_of_CNN), you lose the comparison of different channel.

        outputs = self.reshape_1(outputs)
        outputs = self.lin(outputs)

        outputs = self.reshape_2(outputs)
        check_shape[0] = outputs.shape
        print("lin_reshape Hello world" * 10)
        print(check_shape)
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

        self.FFN_units = FFN_units
        self.l2_rate = l2_rate
        self.embedding_1 = EmbeddingLayer(
            d_model, filters, kernel_size[0], strides[0], l2_rate, name='embedding_1')
        self.embedding_2 = EmbeddingLayer(
            d_model, filters, kernel_size[1], strides[1], l2_rate, name='embedding_2')

        # self.embedding_3 = EmbeddingLayer_wavelet(
        #     d_model, filters, kernel_size[1], strides[1], name='embedding_3')

        input_channel = inputs_shape[1]
        kernel_size_1 = kernel_size[0]
        strides_1 = strides[0]
        patch_1 = (input_channel - kernel_size_1[0]) // strides_1[0] + 1

        kernel_size_2 = kernel_size[1]
        strides_2 = strides[1]
        patch_2 = (input_channel - kernel_size_2[0]) // strides_2[0] + 1

        # patch_3 = input_channel

        self.cls_token_patch_1 = tf.Variable(tf.random.normal((1, 1)))
        self.pos_embedding_patch_1 = tf.Variable(
            tf.random.normal((1, patch_1 * d_model + 1)))
        self.dropout_patch_1 = layers.Dropout(dropout_rate, name='dropout_1')

        self.cls_token_patch_2 = tf.Variable(tf.random.normal((1, 1)))
        self.pos_embedding_patch_2 = tf.Variable(
            tf.random.normal((1, patch_2 * d_model + 1)))
        self.dropout_patch_2 = layers.Dropout(dropout_rate, name='dropout_2')

        # self.cls_token_patch_3 = tf.Variable(tf.random.normal((1, 1, d_model)))
        # self.pos_embedding_patch_3 = tf.Variable(
        #     tf.random.normal((1, patch_3+1, d_model)))
        # self.dropout_patch_3 = layers.Dropout(dropout_rate, name='dropout_3')

        self.encoder_1 = TFBertModel.from_pretrained('bert-base-uncased')
        # Encoder(n_layers,
        #                          FFN_units,
        #                          n_heads,
        #                          dropout_rate,
        #                          activation,
        #                          name="encoder_1")

        self.encoder_2 = TFBertModel.from_pretrained('bert-base-uncased')
        # Encoder(n_layers,
        #                          FFN_units,
        #                          n_heads,
        #                          dropout_rate,
        #                          activation,
        #                          name="encoder_2")

        # self.encoder_3 = Encoder(n_layers,
        #                          FFN_units,
        #                          n_heads,
        #                          dropout_rate,
        #                          activation,
        #                          name="encoder_2")

        # self.global_average_pooling_1 = layers.GlobalAveragePooling1D(
        #     data_format='channels_first', keepdims=False)
        # self.global_average_pooling_2 = layers.GlobalAveragePooling1D(
        #     data_format='channels_first', keepdims=False)
        # self.global_average_pooling_3 = layers.GlobalAveragePooling1D(
        #     data_format='channels_first', keepdims=False)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

        self.last_dense_layers = []
        for i in range(num_of_last_dense):
            self.last_dense_layers.append(layers.Dense(
                self.FFN_units/(2**i), activation=activation, kernel_regularizer=tf.keras.regularizers.l2(self.l2_rate)))
        self.lin = layers.Dense(num_class, activation='softmax')

    def call(self, inputs):
        emb_output_1 = self.embedding_1(inputs)
        emb_output_2 = self.embedding_2(inputs)

        emb_output_1 = tf.squeeze(emb_output_1, axis=1)
        emb_output_2 = tf.squeeze(emb_output_2, axis=1)
        # emb_output_3 = self.embedding_3(inputs)

        # repeat and expand the size to be the same of batch size
        cls_token_patch_tiled_1 = tf.tile(
            self.cls_token_patch_1, [tf.shape(inputs)[0], 1])
        pos_embedding_path_tiled_1 = tf.tile(self.pos_embedding_patch_1, [
                                             tf.shape(inputs)[0], 1])

        cls_token_patch_tiled_2 = tf.tile(
            self.cls_token_patch_2, [tf.shape(inputs)[0], 1])
        pos_embedding_path_tiled_2 = tf.tile(self.pos_embedding_patch_2, [
                                             tf.shape(inputs)[0], 1])

        # cls_token_patch_tiled_3 = tf.tile(
        #     self.cls_token_patch_3, [tf.shape(inputs)[0], 1, 1])
        # pos_embedding_path_tiled_3 = tf.tile(self.pos_embedding_patch_3, [
        #                                      tf.shape(inputs)[0], 1, 1])

        output_1 = tf.concat([cls_token_patch_tiled_1, emb_output_1], axis=1)
        output_1 = output_1 + pos_embedding_path_tiled_1
        output_1 = self.dropout_patch_1(output_1)
        output_1 = tf.cast(output_1*1e2, tf.int32)
        output_1 = self.encoder_1(output_1)["pooler_output"]
        # output_1 = self.global_average_pooling_1(output_1)

        output_2 = tf.concat([cls_token_patch_tiled_2, emb_output_2], axis=1)
        output_2 = output_2 + pos_embedding_path_tiled_2
        output_2 = self.dropout_patch_2(output_2)
        output_2 = tf.cast(output_2*1e2, tf.int32)
        output_2 = self.encoder_2(output_2)["pooler_output"]
        # output_2 = self.global_average_pooling_2(output_2)

        # output_3 = tf.concat([cls_token_patch_tiled_3, emb_output_3], axis=1)
        # output_3 = output_3 + pos_embedding_path_tiled_3
        # output_3 = self.dropout_patch_3(output_3)
        # output_3 = self.encoder_3(output_3)
        # output_3 = self.global_average_pooling_2(output_3)

        output = tf.concat([output_1, output_2], axis=1)  # , output_3
        output = self.norm(output)

        for i in range(len(self.last_dense_layers)):
            output = self.last_dense_layers[i](output)

        output = self.lin(output)
        return output


class Classifier_Bert():
    def __init__(self, output_directory, callbacks, input_shape, epochs):
        # input_shape = (200, 52, 128, 1)

        self.output_directory = output_directory
        self.callbacks = callbacks
        self.epochs = epochs

        # 随机给定超参数进行训练
        # 32#random.choice([16, 32, 48])  # 128 256
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=50)
        self.callbacks.append(early_stopping)
        self.batch_size = 16  # 32  # random.choice([128]) # 没有影响，不改变模型的结构
        kernel_size_1 = (4, 5)  # 2, 3, 4
        stride_size_1 = (1, 2)
        kernel_size_2 = (1, 5)  # 2: random.randint(2,8)  (2,5 are the best)
        stride_size_2 = (1, 2)
        kernel_size = [kernel_size_1, kernel_size_2]
        stride_size = [stride_size_1, stride_size_2]

        # random.choice([2, 3, 4, 5, 6, 7, 8]) 6,7 are the best
        # random.choice([4, 24])  # random.choice([12, 24, 36])
        output_channel = 2  # random.choice([3, 8, 24]) #24

        # random.choice([64, 256])# 64 #
        d_model = 4  # random.choice([64, 128, 256])

        dropout_rate = 0.4

        # random.choice([4, 12])  # random.randint(10, 12)
        n_layers = 12*4  # random.choice([12, 8, 16])
        FFN_units = 256  # random.choice([64, 128, 256, 512])  # 512, 64, 128,
        n_heads = 4  # random.choice([4, 8])  # 2
        #   # random.choice(['relu', 'gelu'])
        activation = 'gelu'  # random.choice(['relu', 'gelu'])
        # warmup_step random.choice([100,200,300,400,500,1000,2000])
        warmup_step = 200
        # random.choice([0.98, 0.99, 0.999])
        adam_beta_1, adam_beta_2 = 0.9, 0.999
        num_of_last_dense = 2  # random.randint(0, 3)
        l2_rate = 0.001

        # kernel_size_1 = (5, 5)
        # stride_size_1 = (1, 2)
        # kernel_size_2 = (1, 5)
        # stride_size_2 = (1, 2)
        # kernel_size = [kernel_size_1, kernel_size_2]
        # stride_size = [stride_size_1, stride_size_2]
        # output_channel = 3
        # d_model = 64
        # dropout_rate = 0.3
        # n_layers = 4
        # FFN_units = 512
        # n_heads = 4  # why this head less can get very good result (4)
        # activation = 'relu'

        num_class = 2
        # lr = random.choice([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.256])
        # momentum = random.choice([0.0, 0.9, 0.99, 0.999])
        # optimizer = tf.keras.optimizers.RMSprop(
        #     learning_rate=lr, rho=0.9, momentum=momentum, epsilon=1e-07, centered=False)

        # If you change these two hyperparameters, remember to change the  self.hyperparameters

        learning_rate = CustomSchedule(
            d_model * FFN_units * n_layers, warmup_step)

        optimizer = tf.keras.optimizers.AdamW(0.01,
                                              beta_1=adam_beta_1,
                                              beta_2=adam_beta_2,
                                              epsilon=1e-9)

        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
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
                              l2_rate)(inputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.summary()
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

        self.hyperparameters = {
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
            "l2_rate":l2_rate,
            "learning_rate": f"CustomSchedule(d_model * FFN_units * n_layers, {warmup_step})",
            "optimizer": f"tf.keras.optimizers.AdamW(learning_rate, beta_1={adam_beta_1}, beta_2={adam_beta_2}, epsilon=1e-9)"  # "optimizer = tf.keras.optimizers.RMSprop(\
            # learning_rate={lr}, rho=0.9, momentum={momentum}, epsilon=1e-07, centered=False)"#
        }
        print(f'hyperparameters: {self.hyperparameters}')

    def fit(self, data, label, X_test, Y_test):
        start_time = time.time()
        print(f'check_shape = {check_shape[0]}')
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

        print(f'Training time is {duration}')

    def predict(self):
        pass
# model = Transformer('transformer', None, None, (5, 52, 128, 1), 1)


"""
        accuracy - 70%
        # kernel_size_1 = (5, 30)
        # stride_size_1 = (1, 4)
        # kernel_size_2 = (1, 30)
        # stride_size_2 = (1, 4)
        # kernel_size = [kernel_size_1, kernel_size_2]
        # stride_size = [stride_size_1, stride_size_2]
        # output_channel = 8
        # d_model = 256
        # dropout_rate = 0.1
        # n_layers = 4
        # FFN_units = 512
        # n_heads = 8
        # num_class = 2
        # lr = 0.0001
        """


""" 71%
        kernel_size_1 = (5, 5)
        stride_size_1 = (1, 2)
        kernel_size_2 = (1, 5)
        stride_size_2 = (1, 2)
        kernel_size = [kernel_size_1, kernel_size_2]
        stride_size = [stride_size_1, stride_size_2]
        output_channel = 3
        d_model = 256
        dropout_rate = 0.3
        n_layers = 4
        FFN_units = 256
        n_heads = 8
        num_class = 2
        
        lr = 0.0001
#       optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
"""


""" 82%
        batch_size = 128
        kernel_size_1 = (5, 5)
        stride_size_1 = (1, 2)
        kernel_size_2 = (1, 5)
        stride_size_2 = (1, 2)
        kernel_size = [kernel_size_1, kernel_size_2]
        stride_size = [stride_size_1, stride_size_2]
        output_channel = 3
        d_model = 64
        dropout_rate = 0.3
        n_layers = 4
        FFN_units = 512
        n_heads = 4
        num_class = 2
        lr = 0.0001
        activation = 'relu
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
"""

""" 88%
        batch_size = 128
        kernel_size_1 = (5, 5)
        stride_size_1 = (1, 2)
        kernel_size_2 = (1, 5)
        stride_size_2 = (1, 2)
        kernel_size = [kernel_size_1, kernel_size_2]
        stride_size = [stride_size_1, stride_size_2]
        output_channel = 3
        d_model = 64
        dropout_rate = 0.3
        n_layers = 4
        FFN_units = 512
        n_heads = 4 # why this head less can get very good result (4)
        num_class = 2
        lr = 1e-4
        activation = 'relu'        
        learning_rate = CustomSchedule(d_model * FFN_units * n_layers, 200)
        optimizer = tf.keras.optimizers.Adam(learning_rate,
                                            beta_1=0.9,
                                            beta_2=0.98,
                                            epsilon=1e-9)
"""
