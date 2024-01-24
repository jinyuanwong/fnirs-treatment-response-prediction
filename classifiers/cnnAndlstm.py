     # FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

from utils.utils import save_logs
from utils.utils import calculate_metrics

class Classifier_CNNandLSTM:

    def __init__(self, output_directory, result_name, input_shape, nb_classes, nb_epochs,verbose=False,build=True):
        self.output_directory = output_directory
        self.batch_size = 64
        self.nb_epochs = nb_epochs#1500
        self.nb_cell = 200
        self.lr = 1e-3
        self.min_lr = 1e-7
        self.result_name = result_name
        # self.normalizer = normalizer
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init' + '-' + result_name + '.hdf5')

        return

    def build_model(self, input_shape, nb_classes):
        padding = 'valid'

        input_layer = keras.layers.Input((input_shape[0],input_shape[1],1))

        if input_shape[0] < 60: # for italypowerondemand dataset
            padding = 'same'
        # normalizer_layer = self.normalizer(input_layer)

        conv1 = keras.layers.Conv2D(filters=32,kernel_size=[40,4],padding=padding)(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation('relu')(conv1)
        conv1 = keras.layers.AveragePooling2D(pool_size=[6,3])(conv1)

        conv2 = keras.layers.Conv2D(filters=32,kernel_size=[40,4],padding=padding)(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)
        conv2 = keras.layers.AveragePooling2D(pool_size=[6,3])(conv2)

        flatten_layer = keras.layers.Flatten()(conv2)

        #
        flatten_layer_shape = flatten_layer.get_shape().as_list()
        reshape_flatten = keras.layers.Reshape(((flatten_layer_shape[1]),1))(flatten_layer)

        lstm_layer = keras.layers.LSTM(self.nb_cell)(reshape_flatten)
        dropout_layer = keras.layers.Dropout(0.5)(lstm_layer)

        output_layer = keras.layers.Dense(units=1,activation='relu')(dropout_layer)#

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(self.lr),
                      metrics=['accuracy'])
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50,
                                                      min_lr=self.min_lr)

        file_path = self.output_directory + 'best_model' + '-' + self.result_name + '.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr,model_checkpoint]

        return model

    def fit(self,x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        # x_val and y_val are only used to monitor the test loss and NOT for training

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory+'last_model.hdf5')

        model = keras.models.load_model(self.output_directory + 'best_model' + '-' + self.result_name + '.hdf5')

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, self.result_name, hist, y_pred, y_true, duration,lr=False)

        keras.backend.clear_session()

    def predict(self, x_test,y_true,x_train,y_train,y_test,return_df_metrics = True):
        model_path = self.output_directory + 'best_model' + '-' + self.result_name + '.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred
