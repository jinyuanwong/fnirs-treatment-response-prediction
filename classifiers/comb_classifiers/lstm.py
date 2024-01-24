
import time
from utils.utils import calculate_metrics
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.optimizers as ko
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils.utils_mine import*
from utils.utils import save_logs
import tensorflow.keras as keras


class Classifier_LSTM:

    def __init__(self, output_directory, result_name, input_shape, nb_classes, nb_epochs,hyperparameter,useCombinationModel,verbose=False,build=True):
        self.output_directory = output_directory
        self.batch_size = hyperparameter['batch_size']
        self.nb_epochs = nb_epochs#1500
        self.lr = hyperparameter['lr']
        self.min_lr = hyperparameter['min_lr']
        self.kernel_size = hyperparameter['kernel_size']
        self.padding = hyperparameter['padding']
        self.activation = hyperparameter['activation']
        self.pool_size = hyperparameter['pool_size']
        self.cell = hyperparameter['cell']
        self.Dropout_rate = hyperparameter['Dropout_rate']
        self.callbacks = hyperparameter['callbacks']
        self.useCombinationModel = useCombinationModel

        if build == True:
            self.result_name = result_name
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init' + '-' + result_name + '.hdf5')

        return
    def LSTM(self,channel):

        m0 = kl.LSTM(self.cell)(channel)
        m0 = kl.Dropout(self.Dropout_rate)(m0)

        if self.useCombinationModel == True:
            m0 = kl.Dense(1, activation=self.activation)(m0)

        m0 = km.Model(inputs=channel, outputs=m0)

        return m0

    def build_model(self, input_shape, nb_classes):

        if self.useCombinationModel == True:

            m0 = self.LSTM(kl.Input(input_shape))
            m1 = self.LSTM(kl.Input(input_shape))
            m2 = self.LSTM(kl.Input(input_shape))
            m3 = self.LSTM(kl.Input(input_shape))
            m4 = self.LSTM(kl.Input(input_shape))
            m5 = self.LSTM(kl.Input(input_shape))
            m6 = self.LSTM(kl.Input(input_shape))
            m7 = self.LSTM(kl.Input(input_shape))
            m8 = self.LSTM(kl.Input(input_shape))
            m9 = self.LSTM(kl.Input(input_shape))
            m10 = self.LSTM(kl.Input(input_shape))
            m11 = self.LSTM(kl.Input(input_shape))
            m12 = self.LSTM(kl.Input(input_shape))
            m13 = self.LSTM(kl.Input(input_shape))
            m14 = self.LSTM(kl.Input(input_shape))
            m15 = self.LSTM(kl.Input(input_shape))
            m16 = self.LSTM(kl.Input(input_shape))
            m17 = self.LSTM(kl.Input(input_shape))
            m18 = self.LSTM(kl.Input(input_shape))
            m19 = self.LSTM(kl.Input(input_shape))
            m20 = self.LSTM(kl.Input(input_shape))
            m21 = self.LSTM(kl.Input(input_shape))
            m22 = self.LSTM(kl.Input(input_shape))
            m23 = self.LSTM(kl.Input(input_shape))
            m24 = self.LSTM(kl.Input(input_shape))
            m25 = self.LSTM(kl.Input(input_shape))
            m26 = self.LSTM(kl.Input(input_shape))
            m27 = self.LSTM(kl.Input(input_shape))
            m28 = self.LSTM(kl.Input(input_shape))
            m29 = self.LSTM(kl.Input(input_shape))
            m30 = self.LSTM(kl.Input(input_shape))
            m31 = self.LSTM(kl.Input(input_shape))
            m32 = self.LSTM(kl.Input(input_shape))
            m33 = self.LSTM(kl.Input(input_shape))
            m34 = self.LSTM(kl.Input(input_shape))
            m35 = self.LSTM(kl.Input(input_shape))
            m36 = self.LSTM(kl.Input(input_shape))
            m37 = self.LSTM(kl.Input(input_shape))
            m38 = self.LSTM(kl.Input(input_shape))
            m39 = self.LSTM(kl.Input(input_shape))
            m40 = self.LSTM(kl.Input(input_shape))
            m41 = self.LSTM(kl.Input(input_shape))
            m42 = self.LSTM(kl.Input(input_shape))
            m43 = self.LSTM(kl.Input(input_shape))
            m44 = self.LSTM(kl.Input(input_shape))
            m45 = self.LSTM(kl.Input(input_shape))
            m46 = self.LSTM(kl.Input(input_shape))
            m47 = self.LSTM(kl.Input(input_shape))
            m48 = self.LSTM(kl.Input(input_shape))
            m49 = self.LSTM(kl.Input(input_shape))
            m50 = self.LSTM(kl.Input(input_shape))
            m51 = self.LSTM(kl.Input(input_shape))

            combined = kl.concatenate(
                [m0.output, m1.output, m2.output, m3.output, m4.output, m5.output, m6.output, m7.output, m8.output,
                 m9.output, m10.output, m11.output, m12.output, m13.output, m14.output, m15.output, m16.output, m17.output,
                 m18.output, m19.output, m20.output, m21.output, m22.output, m23.output, m24.output, m25.output, m26.output,
                 m27.output, m28.output, m29.output, m30.output, m31.output, m32.output, m33.output, m34.output, m35.output,
                 m36.output, m37.output, m38.output, m39.output, m40.output, m41.output, m42.output, m43.output, m44.output,
                 m45.output, m46.output, m47.output, m48.output, m49.output, m50.output, m51.output])

            z = kl.Dense(64, activation=self.activation)(combined)
            z = kl.Dropout(rate=0.25)(z)
            # z = kl.Dense(32, activation=self.activation)(z)
            # z = kl.Dropout(rate=0.25)(z)
            # z = kl.Dense(16, activation=self.activation)(z)
            z = kl.Dense(2, activation="softmax")(z)

            model = km.Model(
                inputs=[m0.input, m1.input, m2.input, m3.input, m4.input, m5.input, m6.input, m7.input, m8.input, m9.input,
                        m10.input, m11.input, m12.input, m13.input, m14.input, m15.input, m16.input, m17.input, m18.input,
                        m19.input, m20.input, m21.input, m22.input, m23.input, m24.input, m25.input, m26.input, m27.input,
                        m28.input, m29.input, m30.input, m31.input, m32.input, m33.input, m34.input, m35.input, m36.input,
                        m37.input, m38.input, m39.input, m40.input, m41.input, m42.input, m43.input, m44.input, m45.input,
                        m46.input, m47.input, m48.input, m49.input, m50.input, m51.input], outputs=z)
        else:
            m0 = self.LSTM(kl.Input(input_shape))
            m = kl.Dense(2, activation='softmax')(m0.output)
            model = km.Model(inputs=m0.input, outputs=m)

        model.compile(optimizer=ko.Adam(self.lr), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def fit(self,X_train, Y_train, X_val, Y_val,X_test,Y_test):


        # x_val and y_val are only used to monitor the test loss and NOT for training

        start_time = time.time()
        if self.useCombinationModel == True:
            hist = self.model.fit(
                [X_train[:, :, 0], X_train[:, :, 1], X_train[:, :, 2], X_train[:, :, 3], X_train[:, :, 4], X_train[:, :, 5],
                 X_train[:, :, 6], X_train[:, :, 7], X_train[:, :, 8], X_train[:, :, 9], X_train[:, :, 10],
                 X_train[:, :, 11], X_train[:, :, 12], X_train[:, :, 13], X_train[:, :, 14], X_train[:, :, 15],
                 X_train[:, :, 16], X_train[:, :, 17], X_train[:, :, 18], X_train[:, :, 19], X_train[:, :, 20],
                 X_train[:, :, 21], X_train[:, :, 22], X_train[:, :, 23], X_train[:, :, 24], X_train[:, :, 25],
                 X_train[:, :, 26], X_train[:, :, 27], X_train[:, :, 28], X_train[:, :, 29], X_train[:, :, 30],
                 X_train[:, :, 31], X_train[:, :, 32], X_train[:, :, 33], X_train[:, :, 34], X_train[:, :, 35],
                 X_train[:, :, 36], X_train[:, :, 37], X_train[:, :, 38], X_train[:, :, 39], X_train[:, :, 40],
                 X_train[:, :, 41], X_train[:, :, 42], X_train[:, :, 43], X_train[:, :, 44], X_train[:, :, 45],
                 X_train[:, :, 46], X_train[:, :, 47], X_train[:, :, 48], X_train[:, :, 49], X_train[:, :, 50],
                 X_train[:, :, 51]], Y_train, batch_size=self.batch_size, epochs=self.nb_epochs, verbose=False, validation_data=(
                [X_val[:, :, 0], X_val[:, :, 1], X_val[:, :, 2], X_val[:, :, 3], X_val[:, :, 4], X_val[:, :, 5],
                 X_val[:, :, 6], X_val[:, :, 7], X_val[:, :, 8], X_val[:, :, 9], X_val[:, :, 10], X_val[:, :, 11],
                 X_val[:, :, 12], X_val[:, :, 13], X_val[:, :, 14], X_val[:, :, 15], X_val[:, :, 16], X_val[:, :, 17],
                 X_val[:, :, 18], X_val[:, :, 19], X_val[:, :, 20], X_val[:, :, 21], X_val[:, :, 22], X_val[:, :, 23],
                 X_val[:, :, 24], X_val[:, :, 25], X_val[:, :, 26], X_val[:, :, 27], X_val[:, :, 28], X_val[:, :, 29],
                 X_val[:, :, 30], X_val[:, :, 31], X_val[:, :, 32], X_val[:, :, 33], X_val[:, :, 34], X_val[:, :, 35],
                 X_val[:, :, 36], X_val[:, :, 37], X_val[:, :, 38], X_val[:, :, 39], X_val[:, :, 40], X_val[:, :, 41],
                 X_val[:, :, 42], X_val[:, :, 43], X_val[:, :, 44], X_val[:, :, 45], X_val[:, :, 46], X_val[:, :, 47],
                 X_val[:, :, 48], X_val[:, :, 49], X_val[:, :, 50], X_val[:, :, 51]], Y_val), callbacks=self.callbacks)
        else:
            hist = self.model.fit(
                X_train, Y_train,
                batch_size=self.batch_size, epochs=self.nb_epochs, verbose=False,
                validation_data=(X_val, Y_val),
                callbacks=self.callbacks
            )
        duration = time.time() - start_time

        # self.model.save(self.output_directory+'last_model.hdf5')

        model = km.load_model(self.output_directory + 'best_model' + '-' + self.result_name + '.hdf5')

        if self.useCombinationModel == True:

            Y_pred = model.predict([X_test[:, :, 0], X_test[:, :, 1], X_test[:, :, 2], X_test[:, :, 3], X_test[:, :, 4], X_test[:, :, 5],
                 X_test[:, :, 6], X_test[:, :, 7], X_test[:, :, 8], X_test[:, :, 9], X_test[:, :, 10], X_test[:, :, 11],
                 X_test[:, :, 12], X_test[:, :, 13], X_test[:, :, 14], X_test[:, :, 15], X_test[:, :, 16], X_test[:, :, 17],
                 X_test[:, :, 18], X_test[:, :, 19], X_test[:, :, 20], X_test[:, :, 21], X_test[:, :, 22], X_test[:, :, 23],
                 X_test[:, :, 24], X_test[:, :, 25], X_test[:, :, 26], X_test[:, :, 27], X_test[:, :, 28], X_test[:, :, 29],
                 X_test[:, :, 30], X_test[:, :, 31], X_test[:, :, 32], X_test[:, :, 33], X_test[:, :, 34], X_test[:, :, 35],
                 X_test[:, :, 36], X_test[:, :, 37], X_test[:, :, 38], X_test[:, :, 39], X_test[:, :, 40], X_test[:, :, 41],
                 X_test[:, :, 42], X_test[:, :, 43], X_test[:, :, 44], X_test[:, :, 45], X_test[:, :, 46], X_test[:, :, 47],
                 X_test[:, :, 48], X_test[:, :, 49], X_test[:, :, 50], X_test[:, :, 51]])

        else:
            Y_pred = model.predict(X_test)

        # convert the predicted from binary to integer
        Y_pred = np.argmax(Y_pred, axis=1)
        Y_true = np.argmax(Y_test, axis=1)

        save_logs(self.output_directory, self.result_name, hist, Y_pred, Y_true, duration,lr=False)

        keras.backend.clear_session()

    def predict(self, X_test,Y_test,x_train,y_train,y_test,return_df_metrics = True):
        model_path = self.output_directory + 'best_model' + '-' + self.result_name + '.hdf5'
        model = keras.models.load_model(model_path)
        Y_true = np.argmax(Y_test, axis=1)

        if self.useCombinationModel == True:
            y_pred = model.predict(
                [X_test[:, :, 0], X_test[:, :, 1], X_test[:, :, 2], X_test[:, :, 3], X_test[:, :, 4], X_test[:, :, 5],
                 X_test[:, :, 6], X_test[:, :, 7], X_test[:, :, 8], X_test[:, :, 9], X_test[:, :, 10], X_test[:, :, 11],
                 X_test[:, :, 12], X_test[:, :, 13], X_test[:, :, 14], X_test[:, :, 15], X_test[:, :, 16], X_test[:, :, 17],
                 X_test[:, :, 18], X_test[:, :, 19], X_test[:, :, 20], X_test[:, :, 21], X_test[:, :, 22], X_test[:, :, 23],
                 X_test[:, :, 24], X_test[:, :, 25], X_test[:, :, 26], X_test[:, :, 27], X_test[:, :, 28], X_test[:, :, 29],
                 X_test[:, :, 30], X_test[:, :, 31], X_test[:, :, 32], X_test[:, :, 33], X_test[:, :, 34], X_test[:, :, 35],
                 X_test[:, :, 36], X_test[:, :, 37], X_test[:, :, 38], X_test[:, :, 39], X_test[:, :, 40], X_test[:, :, 41],
                 X_test[:, :, 42], X_test[:, :, 43], X_test[:, :, 44], X_test[:, :, 45], X_test[:, :, 46], X_test[:, :, 47],
                 X_test[:, :, 48], X_test[:, :, 49], X_test[:, :, 50], X_test[:, :, 51]])
        else:
            y_pred = model.predict(X_test)

        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(Y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred
