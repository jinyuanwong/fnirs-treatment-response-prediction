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
        
        args.update_model_checkpoint(output_directory + 'checkpoint')
        self.callbacks.append(args.model_checkpoint)
        self.callbacks.append(args.earlystopping)
        self.callbacks.append(args.reduce_lr)    
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
        
        output_list = []
        # Define outputs
        for metric_name, _ in  args.metrics.items():
            print('Check - order if it is right:', metric_name)
            output_metric = layers.Dense(2, activation='softmax', name=metric_name)(x)
            output_list.append(output_metric)
            
        model = tf.keras.Model(inputs=inputs_time_point, outputs=output_list)
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
        if self.params['args'].load_previous_checkpoint == True:
            if os.path.exists(self.output_directory + 'checkpoint'):
                self.model.load_weights(self.output_directory + 'checkpoint')
        
        y = {metric_name: Y_train[:, i] for i, metric_name in enumerate(self.params['args'].metrics)}
        validation_y = {metric_name: Y_val[:, i] for i, metric_name in enumerate(self.params['args'].metrics)}

        hist = self.model.fit(
            x=X_train,
            y=y,
            validation_data=(X_val, validation_y),
            # y={'gender': Y_train[:, 0], 'age': Y_train[:, 1], 'education': Y_train[:, 2], 'smoking': Y_train[:, 3], 'alcohol': Y_train[:, 4], 'HAMD_Scores': Y_train[:, 5], 'Suicide_Risk': Y_train[:, 6], 'depression': Y_train[:, 7]},
            # validation_data=(X_val, {'gender': Y_val[:, 0], 'age': Y_val[:, 1], 'education': Y_val[:, 2], 'smoking': Y_val[:, 3], 'alcohol': Y_val[:, 4], 'HAMD_Scores': Y_val[:, 5], 'Suicide_Risk': Y_val[:, 6], 'depression': Y_val[:, 7]}),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=True,
            shuffle=True,
            # class_weight=self.class_weights_dict
        )

        self.model.load_weights(self.output_directory + 'checkpoint')
            
        Y_test_pred = self.model.predict(X_test)
        Y_true = np.argmax(Y_test, axis=1)
        Y_val_pred = np.argmax(self.model.predict(X_val), axis=1)
        Y_val_true = np.argmax(Y_val, axis=1)

        duration = time.time() - start_time
        self.info['duration'] = duration
        
        save_validation_acc_multi_task(self.output_directory, self.model.predict(X_val), Y_val, self.params['args'].metrics, self.info)
        save_validation_acc_multi_task(self.output_directory, self.model.predict(X_test), Y_test, self.params['args'].metrics, self.info, save_file_name='test_acc.txt')
        
        # if check_if_save_model(self.output_directory, self.model.predict(X_test), Y_test, self.info['monitor_metric'], self.info):
        #     save_logs(self.model, self.output_directory, None, hist, Y_test_pred, Y_test, duration, lr=True, is_saving_checkpoint=False, hyperparameters=None)

        print(f'Training time is {duration}')
        save_current_file_to_folder(os.path.abspath(__file__), self.output_directory)
        if self.params.get('config_file_path') is not None:
            save_current_file_to_folder(self.params['config_file_path'], self.output_directory)

    def predict(self):
        pass