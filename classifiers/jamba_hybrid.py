import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import math

import numpy as np
import time
from utils.utils_mine import *
from utils.utils import *
from utils.utils_mine import generate_fnirs_adj_tf

from classifiers.model.mamba import ResidualBlock

from classifiers.layer.transformers import Transformer_Encoder
from classifiers.layer.rmsnorm import RMSNorm
from classifiers.model.jamba import Transformer_layer, Attention_MoE_layer, Mamba_layer, Mamba_MoE_layer
from classifiers.layer.mamba import MambaBlock
from classifiers.layer.gnn import GNN
from classifiers.layer.attention import ChannelAttentionLayer
from classifiers.layer.rmsnorm import RMSNorm
from classifiers.layer.conv1d import conv1d_layer

from utils.callbacks import CustomModelCheckpoint

class Classifier_Jamba():
    def __init__(self, output_directory, callbacks, input_shape, epochs, sweep_config, info):

        self.output_directory = output_directory
        self.callbacks = callbacks
        self.epochs = epochs

        self.info = info
        self.params = params = info['parameter']
        args = self.params['args']
        self.class_weights_dict = {0: 1, 1: args.classweight1}
        
        self.callbacks.append(args.earlystopping)
                
        self.batch_size = args.batch_size

        num_class = 2  # 2
        # If you change these two hyperparameters, remember to change the  self.hyperparameters
        
        # if input_shape[-1] != 1 and input_shape[-1] > 10:
        #     inputs_time_point = tf.keras.Input(shape=(input_shape[1:]+[1]))
        # else:
        inputs_time_point = tf.keras.Input(shape=input_shape[1:])

        # Extract the first 52 channels as time series points
        x = inputs_time_point[:, :52, :]

        # Extract the last 52 channels as features
        y = inputs_time_point[:, -52:, :27]     
           
        adj = generate_fnirs_adj_tf() 
        
        conv1d_x = conv1d_layer(args)(x)
        
        x = RMSNorm()(x)
        # x = ChannelAttentionLayer(input_shape[1])(inputs_time_point)
        x = MambaBlock(args)(x)

        x = GNN(args.model_internal_dim, adj, args.activation, args.dropout_rate)(x)
        
        for _ in range(args.num_layers):
            x = Mamba_layer(args)(x)
            x = Mamba_MoE_layer(args)(x)   
        
        y = RMSNorm()(y)
        y = layers.Dense(args.model_internal_dim, activation=tf.nn.gelu)(y)
        y = layers.Dropout(args.dropout_rate*2)(y)
        
        z = tf.concat([x, y], axis=-1)
                    
        z = Transformer_layer(
                            FFN_units=args.model_internal_dim*2,
                            n_heads=args.n_heads,
                            dropout_rate=args.dropout_rate,
                            activation=tf.nn.gelu,
                            )(z)
        z = Attention_MoE_layer(
                            FFN_units=args.model_internal_dim*2,
                            n_heads=args.n_heads,
                            dropout_rate=args.dropout_rate,
                            activation=tf.nn.gelu,
                            n_experts = args.n_experts,
                            )(z)      
        
        z = tf.concat([z, conv1d_x], axis=-1)
        if not args.use_lm_head: 
            z = layers.Flatten()(z)
        z = layers.Dense(args.last_dense_units, activation=tf.nn.gelu)(z)
        
        outputs = layers.Dense(num_class, activation=args.final_activation)(z)
        model = tf.keras.Model(inputs=inputs_time_point, outputs=outputs)
        model.summary()

        optimizer = tf.keras.optimizers.AdamW(args.learning_rate,
                                        beta_1=args.beta_1,
                                        beta_2=args.beta_2,
                                        epsilon=args.epsilon,
                                        clipnorm=args.clipnorm)
        
        model.compile(optimizer=optimizer,
                      loss=args.loss,#categorical_crossentropy
                      metrics=args.metrics) # , Recall(name='sensitivity')

        self.model = model


    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        start_time = time.time()

        model_path = self.output_directory + 'checkpoint'
        if os.path.exists(model_path):
            self.model.load_weights(model_path)        
        
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

        self.model.load_weights(self.output_directory + 'checkpoint')
            
        Y_test_pred = self.model.predict(X_test)
        Y_true = np.argmax(Y_test, axis=1)
        Y_val_pred = np.argmax(self.model.predict(X_val), axis=1)
        Y_val_true = np.argmax(Y_val, axis=1)

        duration = time.time() - start_time
        self.info['duration'] = duration
        
        save_validation_acc(self.output_directory, self.model.predict(X_val), Y_val, self.info['monitor_metric'], self.info)
        save_validation_acc(self.output_directory, self.model.predict(X_test), Y_test, self.info['monitor_metric'], self.info,
                            save_file_name='test_acc.txt')
        
        if check_if_save_model(self.output_directory, self.model.predict(X_test), Y_test, self.info['monitor_metric'], self.info):
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
