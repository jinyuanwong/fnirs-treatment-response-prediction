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

import wandb
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


# def read_past_value(directory, check_metrice):
#     hist_loc = directory + 'history.csv'
#     if os.path.exists(hist_loc):
#         history = pd.read_csv(hist_loc)
#         return np.max(history['val_' + check_metrice])
#     else:
#         return 0


# def read_current_value(Y_pred, Y_true, check_metrice):

#     if check_metrice == 'accuracy':
#         return accuracy_score(Y_true, np.argmax(Y_pred, axis=1))
#     else:
#         raise ('You have not create a calculation for: ' + check_metrice)


# def check_if_save_model(output_directory, Y_pred, Y_true, check_metrice):
#     past_metrice = read_past_value(output_directory, check_metrice)
#     current_metrice = read_current_value(Y_pred, Y_true, check_metrice)
#     print(f"Current accuracy: {current_metrice}")
#     if current_metrice > past_metrice:
#         return True
#     return False


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


class CFNN(tf.keras.Model):
    def __init__(self, activation):
        super(CFNN, self).__init__()
        dense_size = [64, 64, 64, 64, 64, 2]
        self.hidden0 = layers.Dense(dense_size[0], activation=activation)
        self.hidden1 = layers.Dense(dense_size[1], activation=activation)
        self.hidden2 = layers.Dense(dense_size[2], activation=activation)
        self.hidden3 = layers.Dense(dense_size[3], activation=activation)
        self.hidden4 = layers.Dense(dense_size[4], activation=activation)

    def call(self, inputs):
        hidden0 = self.hidden0(inputs)
        hidden1 = self.hidden1(layers.Concatenate()([inputs, hidden0]))
        hidden2 = self.hidden2(layers.Concatenate()([inputs, hidden1]))
        hidden3 = self.hidden3(layers.Concatenate()([inputs, hidden2]))
        hidden4 = self.hidden4(layers.Concatenate()([inputs, hidden3]))
        return hidden4


class Classifier_CFNN():
    def __init__(self, output_directory, callbacks, input_shape, epochs, sweep_config, info):
        # input_shape = (200, 52, 128, 1)

        self.output_directory = output_directory
        self.callbacks = callbacks
        self.epochs = epochs
        parameter = info['parameter']
        self.sweep_config = sweep_config

        # 随机给定超参数进行训练
        # 32#random.choice([16, 32, 48])  # 128 256
        early_stopping = EarlyStopping(monitor='val_loss', patience=100)
        self.info = info
        self.callbacks.append(early_stopping)

        self.batch_size = 128  # sweep_config['batch_size']  # 128

        # sweep_config['activation']  # 10
        activation = parameter['activation']

        # warmup_step random.choice([100,200,300,400,500,1000,2000])
        warmup_step = 200
        # random.choice([0.98, 0.99, 0.999])
        adam_beta_1, adam_beta_2 = 0.9, 0.999
        l2_rate = 0.001
        # learning_rate = CustomSchedule(
        #     d_model_1 * d_model_2 / sweep_config['lr'], warmup_step)
        optimizer = tf.keras.optimizers.Adam(parameter['lr'],
                                             beta_1=adam_beta_1,
                                             beta_2=adam_beta_2,
                                             epsilon=1e-9)

        # If you change these two hyperparameters, remember to change the  self.hyperparameters

        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        input_features = tf.keras.Input(
            shape=input_shape[1:])  # Replace with actual shape
        # Replace with actual shape

        outputs = CFNN(activation)(input_features)
        outputs = layers.Dense(2, activation='softmax')(outputs)
        model = tf.keras.Model(
            inputs=input_features, outputs=outputs)
        model.summary()
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

        self.hyperparameters = parameter
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
        save_validation_acc(self.output_directory, np.argmax(self.model.predict(
            X_val), axis=1), np.argmax(Y_val, axis=1), self.info['monitor_metric'], self.info)
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
