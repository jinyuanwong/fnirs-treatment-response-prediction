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
from classifiers.layer.cls_position_encoding_layer import PositionEncodingLayer
from classifiers.model.transformer import Transformer
from classifiers.model.general_graph_transformer import GraphTransformer

# Transformer was based on
#   Zenghui Wang' Pytorch implementation. https://github.com/wzhlearning/fNIRS-Transformer/blob/main/model.py
# Adapted to Tensorflow by Jinyuan Wang
# MIT License




class Classifier_Graph_Transformer():
    def __init__(self, output_directory, callbacks, input_shape, epochs, sweep_config, info):
        # input_shape = (200, 52, 128, 1)

        self.output_directory = output_directory
        self.callbacks = callbacks
        self.epochs = epochs

        # 随机给定超参数进行训练
        # 32#random.choice([16, 32, 48])  # 128 256
        early_stopping = EarlyStopping(monitor='val_loss', patience=100)
        self.info = info
        params = info['parameter']
        self.callbacks.append(early_stopping)
        self.batch_size = params['batch_size'] if params.get('batch_size') else 128
        d_model = params['d_model'] if params.get('d_model') else 64  # 125# # random.choice([64, 128, 256])
        dropout_rate = 0.4
        n_layers = params['n_layers'] if params.get('n_layers') else 12
        gnn_layers = sweep_config['gnn_layers'] if sweep_config else 1
        FFN_units = sweep_config['FFN_units'] if sweep_config else 256
        n_heads = 1 # specific for graph_transformer for doing graph operation 
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
        if input_shape[-1] != 1 and input_shape[-1] > 10:
            inputs = tf.keras.Input(shape=(input_shape[1:]+[1]))
        else:
            inputs = tf.keras.Input(shape=input_shape[1:])
        
        num_branches = inputs.shape[-1]
        outputs = []
        for i in range(num_branches*2):
            output = layers.Dense(d_model)(inputs[...,i//2])
            output = PositionEncodingLayer(d_model=d_model, dropout_rate=dropout_rate, name=f'CLS_pos_encoding_{i}')(output)
            output = GraphTransformer(
                dropout_rate,
                n_layers,
                FFN_units,
                n_heads,
                activation,
            )(output, input_adj)
            outputs.append(output)

        # output_1 = layers.Dense(d_model)(inputs)
        # output_2 = layers.Dense(d_model)(inputs)
        # output_1 = PositionEncodingLayer(
        #     d_model=d_model, dropout_rate=dropout_rate, name='CLS_pos_encoding_1')(output_1)
        # output_2 = PositionEncodingLayer(
        #     d_model=d_model, dropout_rate=dropout_rate, name='CLS_pos_encoding_2')(output_2)
        # output_1 = GraphTransformer(
        #     dropout_rate,
        #     n_layers,
        #     FFN_units,
        #     n_heads,
        #     activation,
        # )(output_1, input_adj)
        # output_2 = GraphTransformer(
        #     dropout_rate,
        #     n_layers,
        #     FFN_units,
        #     n_heads,
        #     activation,
        # )(output_2, input_adj)
        # outputs = tf.concat([output_1, output_2], axis=1)  #
        
        outputs = tf.concat(outputs, axis=1)  #

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
        self.info['Y_pred_in_test'] = Y_pred
        Y_pred = np.argmax(Y_pred, axis=1)
        Y_true = np.argmax(Y_test, axis=1)
        Y_val_pred = np.argmax(self.model.predict([X_val, adj_val]), axis=1)
        Y_val_true = np.argmax(Y_val, axis=1)

        duration = time.time() - start_time
        save_validation_acc(self.output_directory, np.argmax(self.model.predict(
            [X_val, adj_val]), axis=1), np.argmax(Y_val, axis=1), self.info['monitor_metric'], self.info)
        save_validation_acc(self.output_directory, np.argmax(self.model.predict([X_test, adj_test]), axis=1), np.argmax(Y_test, axis=1), self.info['monitor_metric'], self.info,
                            save_file_name='test_acc.txt')
        if check_if_save_model(self.output_directory, Y_val_pred, Y_val_true, self.info['monitor_metric'], self.info):
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
