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
from xgboost import XGBClassifier
import wandb
from tensorflow.keras.callbacks import EarlyStopping

# Transformer was based on
#   Zenghui Wang' Pytorch implementation. https://github.com/wzhlearning/fNIRS-Transformer/blob/main/model.py
# Adapted to Tensorflow by Jinyuan Wang
# MIT License


class Classifier_XGBoost():
    def __init__(self, output_directory, callbacks, input_shape, epochs, sweep_config, info):
        # input_shape = (200, 52, 128, 1)

        self.output_directory = output_directory
        self.info = info
        # 随机给定超参数进行训练
        # 32#random.choice([16, 32, 48])  # 128 256
        params = {
            'objective': 'binary:logistic',  # for binary classification
            'eval_metric': 'logloss',        # evaluation metric
            'max_depth': np.random.choice([3, 4, 5, 6, 7, 8, 9, 10]),                  # maximum depth of trees
            'learning_rate': np.random.choice([0.01, 0.05, 0.1, 0.2, 0.3]),            # learning rate
            'early_stopping_rounds': np.random.choice([10, 20, 30, 40, 50]),     # early stoping
            'n_estimators': np.random.choice([50, 100, 150, 200, 250, 300, 400, 500])
        }
        self.params = params
        self.model = XGBClassifier(
            objective=params['objective'], max_depth=params['max_depth'], learning_rate=params['learning_rate'], n_estimators=params['n_estimators'])

        self.hyperparameters = params
        print(f'hyperparameters: {self.hyperparameters}')

    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        start_time = time.time()
        hist = self.model.fit(X_train, Y_train, eval_set=[
            (X_train, Y_train), (X_val, Y_val)], early_stopping_rounds=self.params['early_stopping_rounds'])


        Y_pred = self.model.predict(X_test)
        Y_pred = np.argmax(Y_pred, axis=1)
        Y_true = np.argmax(Y_test, axis=1)

        duration = time.time() - start_time
        save_validation_acc(self.output_directory, np.argmax(
            self.model.predict(X_val), axis=1), np.argmax(Y_val, axis=1), self.info['monitor_metric'], self.info)
        if check_if_save_model(self.output_directory, Y_pred, Y_true, self.info['monitor_metric'], self.info):
            # save learning rate as well
            # Can ignore the result name which has beend set as None
            save_logs(self.model, self.output_directory, None,
                      hist, Y_pred, Y_true, duration,
                      lr=True,
                      is_saving_checkpoint=False,
                      hyperparameters=self.hyperparameters,
                      y_true_onehot=Y_test,
                      y_pred_onehot=tf.one_hot(Y_pred, depth=2).numpy(),
                      pass_history=True
                      )

        print(f'Training time is {duration}')

    def predict(self):
        pass
