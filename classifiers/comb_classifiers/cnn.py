import time
from utils.utils import calculate_metrics
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.optimizers as ko
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils.utils_mine import*
from utils.utils import save_logs
import tensorflow.keras as keras
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.applications import vgg19
from sklearn.metrics import auc,accuracy_score,roc_curve,recall_score
import tensorflow_addons as tfa
import random
# input_shape = (125, 20, 1)


"""
Read the past model accuracy from hist
result_name = classifier_name + '-'+ itr+'-' + str(repeat_count)
"""
def read_past_value(directory,name):
    print(f'directory : {directory}')
    if os.path.exists(directory):
        files = os.listdir(directory)
        for file in files: 
            if file[:3] == 'his':
                location = directory + file
                history = pd.read_csv(location)
                print(f'File Location: {location}')
                return np.max(history['val_accuracy']),np.max(history['val_'+name])
    return 0,0

def get_current_auc(y_pred, y_true):
    y_pred_1 = np.argmax(y_pred, axis=1)
    y_pred_max = np.argmax(y_pred,axis=1)
    auc = tf.keras.metrics.AUC()
    auc.update_state(y_true,y_pred_1)
    return accuracy_score(y_true,y_pred_max),auc.result().numpy()

def get_current_f1_score(y_pred, y_true):
    y_pred_max = np.argmax(y_pred,axis=1)
    metric = tf.keras.metrics.F1Score(average='weighted')
    y_true_onehot = tf.one_hot(y_true, depth=2)
    metric.update_state(y_true_onehot, y_pred)
    f1_score = metric.result().numpy()
    return accuracy_score(y_true,y_pred_max),f1_score

def check_if_save_model(output_directory, result_name, Y_pred, Y_true): 
    past_accuracy,past_f1_score = read_past_value(output_directory,'f1_score')
    current_accuracy,current_f1_score = get_current_f1_score(Y_pred,Y_true)
    y_pred_max = np.argmax(Y_pred,axis=1)
    recall = recall_score(Y_true, y_pred_max)
    print(f'Current sensitivity: {recall}')
    print(f' past_accuracy | current_accuracy: {past_accuracy} | {current_accuracy}')
    print(f' past_auc | current_auc: {past_f1_score} | {current_f1_score}')
    if current_f1_score >= past_f1_score and recall > 0.65:
        return True
    return False

class Classifier_CNN:

    def __init__(self, output_directory, result_name, input_shape, nb_classes, nb_epochs,hyperparameter,useCombinationModel,verbose=False,build=True,flag=0):
        self.flag = flag # if flag == 1 means should find the hyperameters
        self.output_directory = output_directory
        self.hyperparameter = hyperparameter
        self.batch_size = hyperparameter['batch_size']
        self.nb_epochs = nb_epochs#1500
        self.lr = hyperparameter['lr']
        self.min_lr = hyperparameter['min_lr']
        self.kernel_size = hyperparameter['kernel_size']
        self.padding = hyperparameter['padding']
        self.activation = hyperparameter['activation']
        self.pool_size = hyperparameter['pool_size']
        self.Dropout_rate = hyperparameter['Dropout_rate']
        self.callbacks = hyperparameter['callbacks']
        self.useCombinationModel = useCombinationModel
        if build == True:
            self.result_name = result_name
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

        return

    def CNN(self, channel):

        m0 = kl.Conv1D(filters=32, kernel_size=self.kernel_size[0], padding=self.padding, kernel_initializer=HeNormal(seed=random.randint(0, 1000)))(channel)
        # m0 = kl.BatchNormalization()(m0)  # Add batch normalization
        m0 = kl.Activation(self.activation)(m0)  # Apply activation function

        m0 = kl.Conv1D(filters=32, kernel_size=self.kernel_size[0], padding=self.padding, kernel_initializer=HeNormal(seed=random.randint(0, 1000)))(m0)
        # m0 = kl.BatchNormalization()(m0)  # Add batch normalization
        m0 = kl.Activation(self.activation)(m0)  # Apply activation function


        # m0 = kl.Conv1D(filters=32, kernel_size=self.kernel_size[0], padding=self.padding, kernel_initializer=initializer)(channel)
        # # m0 = kl.BatchNormalization()(m0)  # Add batch normalization
        # m0 = kl.Activation(self.activation)(m0)  # Apply activation function

        m0 = kl.Conv1D(filters=1, kernel_size=self.kernel_size[1], padding=self.padding, kernel_initializer=HeNormal(seed=random.randint(0, 1000)))(m0)
        # m0 = kl.BatchNormalization()(m0)  # Add batch normalization
        m0 = kl.Activation(self.activation)(m0)  # Apply activation function
        m0 = kl.Flatten()(m0)

        # 2023-04-14 - modified to improve the result 
        if self.useCombinationModel == True:
            m0 = kl.Dense(1, activation=self.activation)(m0)

        # m0 = kl.GlobalAveragePooling1D()(m0)
        # tcn_layer = layers.TCN(nb_filters=64, kernel_size=self.kernel_size[0], dilations=[1, 2, 4, 8], padding=self.padding, return_sequences=False)(x)

        m0 = km.Model(inputs=channel, outputs=m0)

        return m0

    def build_model(self, input_shape, nb_classes):

        if self.useCombinationModel == True:
            all_m = list()
            channelInput = (input_shape[0],1)
            print(channelInput)
            for i in range(input_shape[1]):
                all_m.append(self.CNN(kl.Input(channelInput)))
                
            if input_shape[1] > 1:
                combined = kl.concatenate([i.output for i in all_m])
                z = kl.Dropout(rate=0.5)(combined)
            else: 
                z = all_m[0].output

            z = kl.Dense(32, activation=self.activation,kernel_initializer = HeNormal(seed=random.randint(0, 1000)))(z)
            z = kl.Dense(2, activation="softmax")(z)
            model = km.Model(inputs=[i.input for i in all_m],outputs = z)
            # model.summary()
        else:
            m0 = self.CNN(kl.Input(input_shape))
            m = kl.Dense(2, activation='softmax', name = 'LastLayer')(m0.output)
            model = km.Model(inputs=m0.input, outputs=m)



        model.compile(
            optimizer=ko.Adam(self.lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy',tf.keras.metrics.F1Score(name='f1_score',average='weighted')]#num_classes=2,
            # tf.keras.metrics.Recall(name='sensitivity')
            # ,
            # keras.metrics.Precision(), keras.metrics.Recall(),
            # keras.metrics.SpecificityAtSensitivity(0.5),
            # keras.metrics.SensitivityAtSpecificity(0.5)
        )#,keras.metrics.Precision(), keras.metrics.Recall()

        return model


    def fit(self,data, label,X_test,Y_test):
        # x_val and y_val are only used to monitor the test loss and NOT for training

        start_time = time.time()
        if self.useCombinationModel == True:

            hist = self.model.fit(
                x=[data[:,:,i] for i in range(data.shape[-2])],y=label,
                validation_data=([X_test[:,:,i] for i in range(X_test.shape[-2])],Y_test),
                batch_size=self.batch_size, epochs=self.nb_epochs, verbose=False, callbacks=self.callbacks)#  #validation_split=0.2,
        else:
            # raise Exception("No yet write the code for useCombinationModel == False")
            hist = self.model.fit(
                x=data[:,:,0],y=label,
                validation_data=(X_test[:,:,self.hyperparameter['channel']],Y_test),
                batch_size=self.batch_size, epochs=self.nb_epochs, verbose=False, callbacks=self.callbacks)#  #validation_split=0.2,
        duration = time.time() - start_time

        # self.model.save(self.output_directory+'last_model.hdf5')

        model = km.load_model(self.output_directory + 'best_model.hdf5')

        if self.useCombinationModel == True:
            Y_pred = model.predict([X_test[:,:,i] for i in range(X_test.shape[-2])])

        else:
            Y_pred = model.predict(X_test[:,:,self.hyperparameter['channel']])

        Y_pred_max = np.argmax(Y_pred,axis=1)
        
        # convert the predicted from binary to integer
        Y_true = np.argmax(Y_test, axis=1)
        

        if check_if_save_model(self.output_directory, self.result_name, Y_pred, Y_true):
            save_logs(model, self.output_directory, self.result_name, hist, Y_pred_max, Y_true, duration,lr=False, y_true_onehot=Y_test, y_pred_onehot=Y_pred)

        keras.backend.clear_session()
        return model

    def predict(self, X_test,Y_test,x_train,y_train,y_test,return_df_metrics = True):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        Y_true = np.argmax(Y_test, axis=1)

        if self.useCombinationModel == True:
            y_pred = model.predict(
                [X_test[:,:,i] for i in range(X_test[-2])])
        else:
            y_pred = model.predict(X_test)

        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(Y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred
