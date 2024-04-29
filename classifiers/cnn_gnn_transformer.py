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
from classifiers.layer.embedding_layer import EmbeddingLayer

# Transformer was based on
#   Zenghui Wang' Pytorch implementation. https://github.com/wzhlearning/fNIRS-Transformer/blob/main/model.py
# Adapted to Tensorflow by Jinyuan Wang
# MIT License


# import pywt
"""
put this function into the utils as well: 
but remember that do not modify utils so much.
"""


@keras.saving.register_keras_serializable()
class F1ScoreCalculation(keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1_score = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, Y_true, Y_pred, sample_weight=None):
        # Cast to float to make sure division will be floating-point operation
        Y_true = tf.cast(Y_true, tf.float32)
        Y_pred = tf.cast(Y_pred, tf.float32)
        
        # Calculate Precision and Recall for binary classification
        epsilon = 1e-7  # to avoid division by zero
        
        TP = tf.reduce_sum(Y_true * Y_pred)
        FP = tf.reduce_sum((1 - Y_true) * Y_pred)
        FN = tf.reduce_sum(Y_true * (1 - Y_pred))
        
        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        
        self.f1_score.assign_add(f1)
    def result(self):
        return self.f1_score

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.f1_score.assign(0.0)




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
    def __init__(self, d_model, dropout_rate, name="ClsPositionEncodingLayer"):
        super(ClsPositionEncodingLayer, self).__init__(name=name)

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


class GCN(tf.keras.Model):
    def __init__(self,
                 d_model,
                 activation='relu'):
        super(GCN, self).__init__()

        self.W = layers.Dense(units=d_model)

        if activation == 'relu':
            self.activation = tf.keras.layers.ReLU()
        elif activation == 'sigmoid':
            self.activation = tf.keras.layers.Activation('sigmoid')
        elif activation == 'tanh':
            self.activation = tf.keras.layers.Activation('tanh')
        elif activation == 'prelu':
            self.activation = tf.keras.layers.Activation('prelu')
        else:
            raise ValueError('Provide a valid activation for GNN')

    def normalize_adjacency(self, adj):
        d = tf.reduce_sum(adj, axis=-1)
        d_sqrt_inv = tf.pow(d, -0.5)
        d_sqrt_inv = tf.where(tf.math.is_inf(d_sqrt_inv), 0., d_sqrt_inv)
        d_mat_inv_sqrt = tf.linalg.diag(d_sqrt_inv)
        return tf.matmul(tf.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    def call(self, inputs, adj):
        adj_normalized = self.normalize_adjacency(adj)
        inputs_features = self.W(inputs)
        outputs = tf.linalg.matmul(adj_normalized, inputs_features)
        return self.activation(outputs)


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


class Classifier_GNN_Transformer():
    def __init__(self, output_directory, callbacks, input_shape, epochs, sweep_config, info):
        # input_shape = (200, 52, 128, 1)

        self.output_directory = output_directory
        self.callbacks = callbacks

        # 随机给定超参数进行训练
        # 32#random.choice([16, 32, 48])  # 128 256
        early_stopping = EarlyStopping(monitor='val_loss', patience=100)
        self.info = info
        params = info['parameter']
        self.epochs = params['epochs'] if params.get('epochs') else epochs 

        self.callbacks.append(early_stopping)
        # 32  # random.choice([128]) # 没有影响，不改变模型的结构 # 8 is very bad ~70%
        self.batch_size = params['batch_size'] if params.get('batch_size') else 128

        kernel_size = (1, 5)  # 2: random.randint(2,8)  (2,5 are the best)
        stride_size = (1, 2)
        
        output_channel = 4  # random.choice([3, 8, 24]) # 24
        # random.choice([64, 256])# 64 #
        d_model = params['d_model'] if params.get('d_model') else 64  # 125# # random.choice([64, 128, 256])
        dropout_rate = 0.4
        # random.choice([4, 12])  # random.randint(10, 12)
        n_layers = params['n_layers'] if params.get('n_layers') else 12  # random.choice([12, 8, 16])
        gnn_layers = sweep_config['gnn_layers'] if sweep_config else 1  # random.choice([12, 8, 16])
        
        FFN_units = sweep_config['FFN_units'] if sweep_config else 256 # random.choice([64, 128, 256, 512])  # 512, 64, 128,
        n_heads = 4  # 5  # random.choice([4, 8])  # 2
        #   # random.choice(['relu', 'gelu'])
        activation = 'gelu'  # random.choice(['relu', 'gelu'])
        # warmup_step random.choice([100,200,300,400,500,1000,2000])
        warmup_step = 200
        # random.choice([0.98, 0.99, 0.999])
        adam_beta_1, adam_beta_2 = 0.9, 0.999
        num_of_last_dense = 2  # random.randint(0, 3)
        parameter = self.info['parameter']
        l1_rate = parameter['l1_rate']
        l2_rate = parameter['l2_rate']
        num_class = 2  # 2
        
        self.class_weights = {0: 1,  # weight for class 0
                 1: parameter['classweight1']}  # weight for class 1, assuming this is the minority class

        
        lr_factor = self.info['parameter']['lr_factor'] if self.info['parameter'].get('lr_factor') else 1
        learning_rate = CustomSchedule(
            d_model * FFN_units * n_layers * lr_factor, warmup_step)
        optimizer = tf.keras.optimizers.AdamW(learning_rate,
                                              beta_1=adam_beta_1,
                                              beta_2=adam_beta_2,
                                              epsilon=1e-9)

        # If you change these two hyperparameters, remember to change the  self.hyperparameters
        input_adj = tf.keras.Input(shape=(input_shape[1], input_shape[1]))
        inputs = tf.keras.Input(shape=input_shape[1:])
        if input_shape[-1] != 1 and input_shape[-1] > 10:
            inputs = tf.keras.Input(shape=(input_shape[1:]+[1]))
        else:
            inputs = tf.keras.Input(shape=input_shape[1:])
        
        num_branches = inputs.shape[-1]
        outputs = []
        for i in range(num_branches*2):
            output = EmbeddingLayer(
                d_model, output_channel, kernel_size, stride_size, l2_rate, name=f'cnn_embedding_{i+1}')(inputs[..., i//2:i//2+1])

            output = GCN(d_model=d_model)(output, input_adj)
            
            for i in range(1,gnn_layers):
                output = GCN(d_model=d_model)(output, input_adj)
            output = ClsPositionEncodingLayer(d_model=d_model, dropout_rate=dropout_rate, name=f'CLS_pos_encoding_{i}')(output)
            output = Transformer(dropout_rate,
                                n_layers,
                                FFN_units,
                                n_heads,
                                activation)(output)
            outputs.append(output)
        outputs = tf.concat(outputs, axis=1)  #

        outputs = layers.LayerNormalization(epsilon=1e-6)(outputs)

        "Doing this in here is to get the layer[-2] feature"
        for i in range(num_of_last_dense):
            outputs = layers.Dense(FFN_units/(2**i),
                                   activation=activation,
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_rate, l2=l2_rate))(outputs)
        outputs = layers.Dense(num_class, activation='softmax')(outputs)
        model = tf.keras.Model(inputs=[inputs, input_adj], outputs=outputs)
        model.summary()

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',#categorical_crossentropy
                      metrics=['accuracy']) # , Recall(name='sensitivity')
        


        self.model = model

        self.hyperparameters = {
            "Test": 'Adding CLS and using Traditional Position encoding (10000)and CNN encoding',
            "batch_size": self.batch_size,
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

    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test):
        start_time = time.time()

        hist = self.model.fit(
            x=[X_train, adj_train],
            y=Y_train,
            validation_data=([X_val, adj_val], Y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=True,
            shuffle=True,  # Set shuffle to True
            class_weight=self.class_weights 
        )

        self.model.load_weights(
            self.output_directory + 'checkpoint')
        Y_pred = self.model.predict([X_test, adj_test])
        self.info['Y_pred_in_test'] = Y_pred
        Y_pred = np.argmax(Y_pred, axis=1)
        Y_true = np.argmax(Y_test, axis=1)

        duration = time.time() - start_time
        
        save_validation_acc(self.output_directory, np.argmax(self.model.predict([X_val, adj_val]), axis=1), np.argmax(Y_val, axis=1), self.info['monitor_metric'], self.info)

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
