import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import time
from utils.utils_mine import *
from utils.utils import *
from classifiers.model.mamba import ResidualBlock
from classifiers.layer.mamba import MambaBlock


class Classifier_Mamba():
    def __init__(self, output_directory, callbacks, input_shape, epochs, sweep_config, info):

        self.output_directory = output_directory
        self.callbacks = callbacks
        self.epochs = epochs

        self.info = info
        self.params = params = info['parameter']
        args = self.params['args']
        earlystopping = args.earlystopping

        self.class_weights_dict = {0: 1, 1: args.classweight1}
        
        self.callbacks.append(earlystopping)
        self.batch_size = args.batch_size

        num_class = 2  # 2
        # If you change these two hyperparameters, remember to change the  self.hyperparameters
        
        # if input_shape[-1] != 1 and input_shape[-1] > 10:
        #     inputs_time_point = tf.keras.Input(shape=(input_shape[1:]+[1]))
        # else:
        inputs_time_point = tf.keras.Input(shape=input_shape[1:])
        
        # inputs_time_point = tf.keras.Input(shape=input_shape[1:])
        x = MambaBlock(args)(inputs_time_point)
        # layers.Dense(args.model_internal_dim, activation=tf.nn.gelu)(inputs_time_point)
        for i in range(args.num_layers):
            x = ResidualBlock(args, name=f"Residual_{i}")(x)
            x = layers.Dropout(args.dropout_rate)(x) # for regularization
        # x = layers.Permute((2, 1))(x)
        # x = layers.GlobalAveragePooling1D()(x)
        x = layers.LayerNormalization(epsilon=1e-5)(x) # normalization layer
        if not args.use_lm_head: 
            x = layers.Flatten()(x)
            
        x = layers.Dense(args.last_dense_units, activation=tf.nn.gelu)(x)
        
        outputs = layers.Dense(num_class, activation=args.final_activation)(x)
        model = tf.keras.Model(inputs=inputs_time_point, outputs=outputs)
        model.summary()

        optimizer = tf.keras.optimizers.AdamW(args.learning_rate,
                                        beta_1=args.beta_1,
                                        beta_2=args.beta_2,
                                        epsilon=args.epsilon)
        
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
