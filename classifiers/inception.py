import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.utils import save_test_duration


class Classifier_INCEPTION:

    def __init__(self, output_directory, result_name, input_shape, nb_classes, nb_epochs, verbose=False, build=True,):#1500

        self.output_directory = output_directory
        self.batch_size = 16
        self.nb_epochs = nb_epochs#1500
        self.lr = 1e-3
        self.min_lr = 1e-8
        self.nb_filters = 32
        self.use_residual = True
        self.depth = 6
        self.kernel_size_s = [[3,3], [4,4], [5,5], [6,6],[10,3]]
        self.callbacks = None
        self.bottleneck_size = 32
        self.verbose = verbose
        self.result_name = result_name
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.model.save_weights(self.output_directory + 'model_init' + '-' + result_name + '.hdf5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):
        # # bottleneck -> 将通道数限制至32通道
        # if int(input_tensor.shape[-1]) > self.bottleneck_size:
        #     # input_tensor = keras.layers.Input((input_tensor[0],input_tensor[1],1))
        #     input_inception = keras.layers.Conv2D(filters=self.bottleneck_size, kernel_size=[1,1],
        #                                           padding='same', activation=activation, use_bias=False)(input_tensor)
        # else:
        #     # input_tensor = keras.layers.Input((input_tensor[0], input_tensor[1], 1))
        input_inception = input_tensor


        kernel_size_s = self.kernel_size_s

        conv_list = []
        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv2D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))
        max_pool_1 = keras.layers.MaxPool2D(pool_size=[3,3], strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv2D(filters=self.nb_filters, kernel_size=[1,1],
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=3)(conv_list)
        # x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='sigmoid')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv2D(filters=int(out_tensor.shape[-1]), kernel_size=[1,1],
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input((input_shape[0],input_shape[1],1))
        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling2D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(self.lr),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=self.min_lr)

        file_path = self.output_directory + 'best_model'+ '-' + self.result_name + '.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self,x_train, y_train, x_val, y_val, y_true):
        # if not tf.test.is_gpu_available:
        #     print('error no gpu')
        #     exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory, self.result_name, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model' + '-' + self.result_name + '.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred
