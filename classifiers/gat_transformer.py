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
from classifiers.model.GAT import GAT

# Transformer was based on
#   Zenghui Wang' Pytorch implementation. https://github.com/wzhlearning/fNIRS-Transformer/blob/main/model.py
# Adapted to Tensorflow by Jinyuan Wang
# MIT License


# import pywt
"""
put this function into the utils as well: 
but remember that do not modify utils so much.
"""



class Classifier_GAT_Transformer():
    def __init__(self, output_directory, callbacks, input_shape, epochs, sweep_config, info):
        # input_shape = (200, 52, 128, 1)

        self.output_directory = output_directory
        self.callbacks = callbacks
        self.epochs = epochs

        # 随机给定超参数进行训练
        # 32#random.choice([16, 32, 48])  # 128 256
        early_stopping = EarlyStopping(monitor='val_loss', patience=100)
        self.info = info
        parameter = info['parameter']
        self.callbacks.append(early_stopping)
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
        output_1 = GAT(units=d_model, num_layers=parameter['gnn_layers'])(inputs, input_adj)
        output_2 = GAT(units=d_model, num_layers=parameter['gnn_layers'])(inputs, input_adj)

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
        Y_pred = np.argmax(Y_pred, axis=1)
        Y_true = np.argmax(Y_test, axis=1)

        duration = time.time() - start_time

        save_validation_acc(self.output_directory, np.argmax(self.model.predict(
            [X_val, adj_val]), axis=1), np.argmax(Y_val, axis=1), self.info['monitor_metric'], self.info)

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
