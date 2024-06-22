import numpy as np
import tensorflow as tf
import time
from utils.utils_mine import *
from utils.utils import *
import random
import os
import pandas as pd
import math
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import wandb

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
        self.params = params = info['parameter']
        self.model = CatBoostClassifier(iterations=self.params['iterations'], learning_rate=self.params['learning_rate'], depth=self.params['depth'], verbose=0)
        self.hyperparameters = self.params
        print(f'hyperparameters: {self.hyperparameters}')

    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        start_time = time.time()
        Y_train = np.argmax(Y_train, axis=1)
        Y_val = np.argmax(Y_val, axis=1)
        Y_test = np.argmax(Y_test, axis=1)
        hist = self.model.fit(X_train, Y_train, eval_set=(X_val, Y_val))

        Y_pred = self.model.predict(X_test)
        self.info['Y_pred_in_test'] = Y_pred
        # Y_pred = np.argmax(Y_pred, axis=1)
        # Y_true = np.argmax(Y_test, axis=1)

        duration = time.time() - start_time
        save_validation_acc(self.output_directory, self.model.predict(X_val), Y_val, self.info['monitor_metric'], self.info)
        if check_if_save_model(self.output_directory, Y_pred, Y_test, self.info['monitor_metric'], self.info):
            # save learning rate as well
            # Can ignore the result name which has beend set as None
            save_logs(self.model, self.output_directory, None,
                      hist, Y_pred, Y_test, duration,
                      lr=True,
                      is_saving_checkpoint=False,
                      hyperparameters=self.hyperparameters,
                      y_true_onehot=np.eye(2)[Y_test],
                      y_pred_onehot=np.eye(2)[Y_pred],
                      pass_history=True
                      )
        print(f'Training time is {duration}')
        save_current_file_to_folder(os.path.abspath(__file__), self.output_directory)
        if self.params.get('config_file_path') is not None:
            save_current_file_to_folder(self.params['config_file_path'], self.output_directory)

    def predict(self):
        pass
