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

from utils.utils_mine import CustomSchedule
from classifiers.layer.cls_position_encoding_layer import ClsPositionEncodingLayer
from classifiers.model.transformer import Transformer
from classifiers.model.gin import GINModel

# Transformer was based on
#   Zenghui Wang' Pytorch implementation. https://github.com/wzhlearning/fNIRS-Transformer/blob/main/model.py
# Adapted to Tensorflow by Jinyuan Wang
# MIT License


# import pywt
"""
put this function into the utils as well: 
but remember that do not modify utils so much.
"""


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


class Classifier_GIN_Transformer():
    def __init__(self, output_directory, callbacks, input_shape, epochs, sweep_config, info):
        # input_shape = (200, 52, 128, 1)

        self.output_directory = output_directory
        self.callbacks = callbacks
        self.epochs = epochs

        # 随机给定超参数进行训练
        # 32#random.choice([16, 32, 48])  # 128 256
        earlystopping = EarlyStopping(monitor='val_loss', patience=100)
        self.info = info
        self.callbacks.append(earlystopping)
        self.batch_size = 128
        d_model = 64  # 125# # random.choice([64, 128, 256])
        dropout_rate = 0.4
        n_layers = sweep_config['n_layers'] if sweep_config else 12
        gnn_layers = sweep_config['gnn_layers'] if sweep_config else 1
        FFN_units = sweep_config['FFN_units'] if sweep_config else 256
        n_heads = 4
        activation = 'gelu'  # random.choice(['relu', 'gelu'])
        warmup_step = 200
        adam_beta_1, adam_beta_2 = 0.9, 0.999
        num_of_last_dense = 2  # random.randint(0, 3)
        l2_rate = 0.001
        num_class = 2  # 2
        lr_factor = sweep_config['lr_factor'] if sweep_config else 1
        learning_rate = CustomSchedule(
            d_model * FFN_units * n_layers * lr_factor, warmup_step)
        optimizer = tf.keras.optimizers.AdamW(learning_rate,
                                              beta_1=adam_beta_1,
                                              beta_2=adam_beta_2,
                                              epsilon=1e-9)

        inputs = tf.keras.Input(shape=input_shape[1:])
        input_adj = tf.keras.Input(shape=(input_shape[1], input_shape[1]))
        output_1 = GINModel(hidden_dim=d_model)(inputs, input_adj)
        output_2 = GINModel(hidden_dim=d_model)(inputs, input_adj)

        output_1 = ClsPositionEncodingLayer(
            d_model=d_model, dropout_rate=dropout_rate, name='CLS_pos_encoding_1')(output_1)
        output_2 = ClsPositionEncodingLayer(
            d_model=d_model, dropout_rate=dropout_rate, name='CLS_pos_encoding_2')(output_2)

        output_1 = Transformer(
            dropout_rate,
            n_layers,
            FFN_units,
            n_heads,
            activation,
        )(output_1)
        output_2 = Transformer(
            dropout_rate,
            n_layers,
            FFN_units,
            n_heads,
            activation,
        )(output_2)

        outputs = tf.concat([output_1, output_2], axis=1)  #

        outputs = layers.LayerNormalization(epsilon=1e-6)(outputs)

        "Doing this in here is to get the layer[-2] feature"
        for i in range(num_of_last_dense):
            outputs = layers.Dense(FFN_units/(2**i),
                                   activation=activation,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(outputs)
        outputs = layers.Dense(num_class, activation='softmax')(outputs)
        model = tf.keras.Model(inputs=[inputs, input_adj], outputs=outputs)
        model.summary()
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])  # , Recall(name='sensitivity')
        self.model = model

        self.hyperparameters = info['parameter']
        print(f'hyperparameters: {self.hyperparameters}')

    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test):
        start_time = time.time()
        model_path = self.output_directory + 'checkpoint'
        if os.path.exists(model_path):
            self.model.load_weights(model_path)        
        
        hist = self.model.fit(
            x=[X_train, adj_train],
            y=Y_train,
            validation_data=([X_val, adj_val], Y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=True,
            shuffle=True  # Set shuffle to True
        )

        self.model.load_weights(
            self.output_directory + 'checkpoint')
        Y_pred = self.model.predict([X_test, adj_test])
        self.info['Y_pred_in_test'] = Y_pred
        Y_test_pred = self.model.predict([X_test, adj_test])
        Y_true = np.argmax(Y_test, axis=1)
        Y_val_pred = np.argmax(self.model.predict([X_val, adj_val]), axis=1)
        Y_val_true = np.argmax(Y_val, axis=1)

        duration = time.time() - start_time
        self.info['duration'] = duration
        save_validation_acc(self.output_directory, self.model.predict(
            [X_val, adj_val]),Y_val, self.info['monitor_metric'], self.info)
        save_validation_acc(self.output_directory, self.model.predict([X_test, adj_test]), Y_test, self.info['monitor_metric'], self.info,
                            save_file_name='test_acc.txt')
        if check_if_save_model(self.output_directory, self.model.predict([X_test, adj_test]), Y_test, self.info['monitor_metric'], self.info):
            # save learning rate as well
            # Can ignore the result name which has beend set as None
            save_logs(self.model, self.output_directory, None,
                      hist, Y_test_pred, Y_test, duration,
                      lr=True,
                      is_saving_checkpoint=False,
                      hyperparameters=None)

        print(f'Training time is {duration}')
        save_current_file_to_folder(os.path.abspath(__file__), self.output_directory)
        if self.params.get('config_file_path') is not None:
            save_current_file_to_folder(self.params['config_file_path'], self.output_directory)

    def predict(self):
        pass
