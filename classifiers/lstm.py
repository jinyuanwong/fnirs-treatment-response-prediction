import os

from tensorflow import keras
# from keras import backend as K
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Flatten
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate, Activation, GRU, SimpleRNN
from tensorflow.keras.models import Model
from utils.utils import save_logs
from utils.utils import calculate_metrics
import numpy as np

import time

class Classifier_LSTM:
    def __init__(self, output_directory, result_name, input_shape, nb_classes, nb_epochs, verbose=False,build=True):
        self.output_directory = output_directory
        self.batch_size = 16
        self.nb_epochs = nb_epochs#1500
        self.nb_cell = 200
        self.lr = 1e-3
        self.min_lr = 1e-8
        self.result_name = result_name
        if build == True:
            self.model = self.build_model(input_shape,nb_classes)
            if(verbose==True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory+'model_init.hdf5')
        return

    def build_model(self,input_shape,nb_classes):
        print(input_shape)
        ip = Input(shape=input_shape)

        lstm_layer = keras.layers.LSTM(self.nb_cell)(ip)
        dropout_layer = keras.layers.Dropout(0.5)(lstm_layer)

        out = Dense(nb_classes,activation='softmax')(dropout_layer)

        model = Model(ip,out)
        model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(self.lr),metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience = 50, min_lr=self.min_lr)

        file_path = self.output_directory+'best_model' + '-' + self.result_name + '.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
        self.callbacks = [reduce_lr,model_checkpoint]

        model.summary()
        return model


    def fit(self,x_train,y_train,x_val,y_val,y_true):

        start_time = time.time()
        hist = self.model.fit(x_train,y_train,batch_size=self.batch_size,epochs=self.nb_epochs,verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory+'last_model.hdf5')

        model = keras.models.load_model(self.output_directory+'best_model' + '-' + self.result_name + '.hdf5')#学习重复结果

        y_pred = model.predict(x_val)

        y_pred = np.argmax(y_pred , axis=1)

        save_logs(self.output_directory,self.result_name, hist, y_pred, y_true, duration)

        keras.backend.clear_session()


    def predict(self, x_test, y_true,x_train,y_train,y_test,return_df_metrics = True):
        model_path = self.output_directory + 'best_model' + '-' + self.result_name + '.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred
