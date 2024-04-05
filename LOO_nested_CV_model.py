

# 使用当前时间作为随机种子
import sys
import time
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils.utils_mine import *
import tensorflow as tf
import tensorflow.keras as keras
from datetime import date
import numpy as np
import random
import tensorflow_addons as tfa
import config
import gc
from classifiers.classifier_factory import create_classifier
from scripts.plot.DL.read_LOO_nestedCV_gnntr import get_sorted_loo_array
import importlib

current_time = int(time.time())

# set the random seed
random.seed(current_time)
np.random.seed(current_time)
tf.random.set_seed(current_time)

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024*6)])
# 保存日志

preprocessed_hb_fold_path = config.PREPROCESSED_HB_FOLD_PATH
default_hb_fold_path = config.DEFAULT_HB_FOLD_PATH
SPECIFY_FOLD = config.SPECIFY_FOLD
# hbo_fold_path = './allData/Output_npy/twoDoctor/nor-all-hbo-hc-mdd'

# /home/jy/Documents/JinyuanWang_pythonCode/results/wang_alex/HbO-All-HC-MDD


class TrainModel():
    def __init__(self, model_name, config, sweep_config=None):
        self.batch_size = 8
        self.config = config
        self.epochs = config.MAX_EPOCHS
        # ['nor-all-hbo-hc-mdd']  # 'HbO-All-Three'
        self.all_archive = config.INPUT_HB_TYPE
        self.model_name = model_name
        self.parameter = config.PARAMETER[model_name]
        self.all_classifiers = [model_name]
        self.sweep_config = sweep_config
        self.hb_path = self.parameter.get('hb_path')
        self.adj_path = self.parameter.get('adj_path')

    def begin(self, info):

        epochs = self.epochs
        using_adj = self.parameter.get('adj_path')

        for archive in self.all_archive:
            hbo_fold_path = default_hb_fold_path + archive
            fnirs_data_path = preprocessed_hb_fold_path + \
                self.model_name if self.model_name in self.config.MODELS_NEED_PREPROCESS_DATA else hbo_fold_path
            for classifier_name in self.all_classifiers:
                # Read data and split into training+validation and testing set with a ratio of 9:1
                # case - not using adj
                # case using adj include GNN, GNN-Transformer, ....
                if using_adj:
                    data, label, adj = simply_read_data_fnirs(
                        fnirs_data_path, self.model_name, self.hb_path, self.adj_path)
                    self.data, self.label, self.adj = data, label, adj
                else:
                    data, label = simply_read_data_fnirs(
                        fnirs_data_path, self.model_name, self.hb_path, None)
                    self.data, self.label = data, label
                if SPECIFY_FOLD:
                    num_of_k_fold = SPECIFY_FOLD
                else:
                    label_not_one_hot = np.argmax(label, axis=1)
                    num_of_k_fold = 3  # I think 3 will be good as pre-treatment data has 15 positive samples and posttreatment has around 12 positive smaples

                params = info['parameter']
                msg = info['message'] + get_params_info(params)
                loo_array = get_sorted_loo_array(
                    self.model_name, msg, data.shape[0], DATASET=archive)
                print('loo_array', loo_array)
                # loo_array:# range(loo_start_from, data.shape[0]): #
                for current_loo in loo_array:
                    for k in range(num_of_k_fold):
                        if using_adj:
                            X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test = stratified_LOO_nested_CV(
                                data, label, k, num_of_k_fold, current_loo, adj)
                        else:
                            X_train, Y_train, X_val, Y_val, X_test, Y_test = stratified_LOO_nested_CV(
                                data, label, k, num_of_k_fold, current_loo)
                        print(f'X_train: {X_train.shape}')
                        print(f'X_val: {X_val.shape}')
                        print(f'X_test: {X_test.shape}')
                        print(
                            f"total sample size is {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}")

                        # msg = info['message'] + f"d_model_{params['d_model']}_batch_size_{params['batch_size']}_n_layers_{params['n_layers']}"
                        output_directory = os.getcwd() + '/results/' + classifier_name + '/' + \
                            archive + \
                            f'/{msg}/' + \
                            f"LOO_nested_CV/LOO_{current_loo}/stratified_nested_{num_of_k_fold}_CV_fold-{str(k)}" + \
                            '/'

                        checkpoint_path = output_directory + 'checkpoint'

                        def learning_rate_schedule(epoch, learning_rate):
                            return learning_rate

                        lr_monitor = tf.keras.callbacks.LearningRateScheduler(
                            learning_rate_schedule)

                        if self.model_name in ['chao_cfnn', 'zhu_xgboost', 'decision_tree']:
                            input_shape = [self.batch_size,
                                           X_train.shape[1]]
                        elif self.model_name in ['mvg_transformer', 'mgn_transformer', 'mgm_transformer']:
                            input_shape = [self.batch_size,
                                           X_train.shape[1],
                                           X_train.shape[2],
                                           adj_train.shape[-1]]
                        else:
                            input_shape = [self.batch_size] + \
                                list(X_train.shape[1:])

                        model_checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                                           monitor='val_' + self.config.MONITOR_METRIC,
                                                           mode='max',
                                                           save_weights_only=True,
                                                           save_best_only=True)

                        callbacks = [model_checkpoint,
                                     lr_monitor]

                        self.model = create_classifier(
                            classifier_name, output_directory, callbacks, input_shape, epochs, info, self.sweep_config)
                        return

def build_model(model_name, config_file_name, msg):
    config_file_name = 'configs.' + config_file_name
    config = importlib.import_module(config_file_name)

    info = {'current_time_seed': current_time,
            'message': msg,
            'parameter': config.PARAMETER[model_name],
            'monitor_metric': config.MONITOR_METRIC
            }
    
    model = TrainModel(model_name, config=config)
    model.begin(info)
    return model


if __name__ == '__main__':
    arg = sys.argv
    model_name = arg[1]

    info = {'current_time_seed': current_time,
            'message': arg[2],
            'parameter': config.PARAMETER[model_name],
            'monitor_metric': config.MONITOR_METRIC
            }

    config_file_name = 'configs.' + arg[3]
    config = importlib.import_module(config_file_name)
    model = TrainModel(model_name, config=config)
    model.begin()
