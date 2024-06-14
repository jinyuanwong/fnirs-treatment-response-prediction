import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import time
from utils.utils_mine import *
from utils.utils import *
import random
import os
import pandas as pd
import math
from classifiers.model.mamba import ResidualBlock
from types import SimpleNamespace
from typing import Union
from dataclasses import dataclass
# Transformer was based on
#   Zenghui Wang' Pytorch implementation. https://github.com/wzhlearning/fNIRS-Transformer/blob/main/model.py
# Adapted to Tensorflow by Jinyuan Wang
# MIT License


# import pywt
"""
put this function into the utils as well: 
but remember that do not modify utils so much.
"""
# @dataclass
# class ModelArgs:
#     model_input_dims: int = 64
#     model_states: int = 64
#     projection_expand_factor: int = 2
#     conv_kernel_size: int = 4 
#     delta_t_min: float = 0.001 # not used 
#     delta_t_max: float = 0.1 # not used 
#     delta_t_scale: float = 0.1 # not used 
#     delta_t_init_floor: float = 1e-4 # not used 
#     conv_use_bias: bool = True
#     dense_use_bias: bool = False
#     layer_id: int = -1
#     seq_length: int = 128
#     num_layers: int = 5
#     dropout_rate: float = 0.2
#     use_lm_head: float = False
#     num_classes: int = None
#     vocab_size: int = None
#     final_activation = None
#     loss:Union[str, keras.losses.Loss] = None
#     optimizer: Union[str, keras.optimizers.Optimizer] = keras.optimizers.AdamW()
#     metrics = ['accuracy']

#     def __post_init__(self):
#         self.model_internal_dim: int = int(self.projection_expand_factor * self.model_input_dims)

#         self.delta_t_rank = math.ceil(self.model_input_dims/16)
#         if self.layer_id == -1:
#             self.layer_id = np.round(np.random.randint(0, 1000), 4)

#         if self.vocab_size == None:
#             raise ValueError("vocab size cannot be none")

#         if self.use_lm_head:
#             self.num_classes=self.vocab_size
#         else:
#             if self.num_classes == None:
#                 raise ValueError(f'num classes cannot be {self.num_classes}')

#             if self.num_classes == 1:
#                 self.final_activation = 'sigmoid'
#             else:
#                 self.final_activation = 'softmax'

#         if self.loss == None:
#             raise ValueError(f"loss cannot be {self.loss}")

class Classifier_Mamba():
    def __init__(self, output_directory, callbacks, input_shape, epochs, sweep_config, info):

        self.output_directory = output_directory
        self.callbacks = callbacks
        self.epochs = epochs

        early_stopping = EarlyStopping(monitor='val_loss', patience=100)
        self.info = info
        params = info['parameter']
        args = params['args']

        self.class_weights_dict = {0: 1, 1: args.classweight1}
        
        self.callbacks.append(early_stopping)
        self.batch_size = args.batch_size

        num_class = 2  # 2
        lr_begin = args.lr_begin
        
        warmup_step = 200
        adam_beta_1, adam_beta_2 = 0.9, 0.999
        learning_rate = CustomSchedule(lr_begin, warmup_step)
        
        
        # If you change these two hyperparameters, remember to change the  self.hyperparameters
        
        # if input_shape[-1] != 1 and input_shape[-1] > 10:
        #     inputs_time_point = tf.keras.Input(shape=(input_shape[1:]+[1]))
        # else:
        inputs_time_point = tf.keras.Input(shape=input_shape[1:])
        
        # inputs_time_point = tf.keras.Input(shape=input_shape[1:])
        x = layers.Dense(args.model_internal_dim, activation=tf.nn.gelu)(inputs_time_point)
        for i in range(args.num_layers):
            x = ResidualBlock(args, name=f"Residual_{i}")(x)
            x = layers.Dropout(args.dropout_rate)(x) # for regularization


        x = layers.LayerNormalization(epsilon=1e-5)(x) # normalization layer
        if not args.use_lm_head: 
            x = layers.Flatten()(x)
        x = layers.Dense(args.last_dense_units, activation=tf.nn.gelu)(x)
        
        outputs = layers.Dense(num_class, activation=args.final_activation)(x)
        model = tf.keras.Model(inputs=inputs_time_point, outputs=outputs)
        model.summary()

        optimizer = tf.keras.optimizers.AdamW(learning_rate,
                                        beta_1=adam_beta_1,
                                        beta_2=adam_beta_2,
                                        epsilon=1e-9)
        
        model.compile(optimizer=optimizer,
                      loss=args.loss,#categorical_crossentropy
                      metrics=args.metrics) # , Recall(name='sensitivity')

        self.model = model


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
            shuffle=True,  # Set shuffle to True
            class_weight=self.class_weights_dict 
        )

        self.model.load_weights(
            self.output_directory + 'checkpoint')
        Y_pred = self.model.predict(X_test)
        Y_pred = np.argmax(Y_pred, axis=1)
        Y_true = np.argmax(Y_test, axis=1)
        
        Y_val_pred = np.argmax(self.model.predict(X_val), axis=1)
        Y_val_true = np.argmax(Y_val, axis=1)

        duration = time.time() - start_time
        
        save_validation_acc(self.output_directory, np.argmax(self.model.predict(X_val), axis=1), np.argmax(Y_val, axis=1), self.info['monitor_metric'], self.info)
        save_validation_acc(self.output_directory, np.argmax(self.model.predict(X_test), axis=1), np.argmax(Y_test, axis=1), self.info['monitor_metric'], self.info,
                            save_file_name='test_acc.txt')

        if check_if_save_model(self.output_directory, Y_val_pred, Y_val_true, self.info['monitor_metric'], self.info):
            # save learning rate as well
            # Can ignore the result name which has beend set as None
            save_logs(self.model, self.output_directory, None,
                      hist, Y_pred, Y_true, duration,
                      lr=True,
                      is_saving_checkpoint=True,
                      hyperparameters=None,
                      y_true_onehot=Y_test,
                      y_pred_onehot=tf.one_hot(Y_pred, depth=2).numpy()
                      )

        print(f'Training time is {duration}')

    def predict(self):
        pass
