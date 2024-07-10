


# 使用当前时间作为随机种子
from wandb.keras import WandbCallback
import sys
import time
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils.utils_mine import *
from utils.fnirs_utils import augment_data
import tensorflow as tf
import tensorflow.keras as keras
from datetime import date
import numpy as np
import random
import tensorflow_addons as tfa
import wandb
import gc
from classifiers.classifier_factory import create_classifier
from scripts.plot.DL.read_LOO_nestedCV_gnntr import get_sorted_loo_array
import importlib

# current_time = int(time.time())# 1720051797 # 1719981546 # 1719470102# 1719981546# int(time.time()) # 1719919781#

# set the random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024*6)])
# 保存日志


# hbo_fold_path = './allData/Output_npy/twoDoctor/nor-all-hbo-hc-mdd'

# /home/jy/Documents/JinyuanWang_pythonCode/results/wang_alex/HbO-All-HC-MDD

class TrainModel():
    def __init__(self, model_name, config, sweep_config=None):
        self.batch_size = 8
        self.config = config
        self.epochs = config.MAX_EPOCHS
        # ['nor-all-hbo-hc-mdd']  # 'HbO-All-Three'
        self.all_archive = config.INPUT_HB_TYPE
        self.parameter = config.PARAMETER[model_name]
        self.all_classifiers = [model_name]
        self.sweep_config = sweep_config
        self.hb_path = self.parameter.get('hb_path')
        self.adj_path = self.parameter.get('adj_path')
        self.label_path = self.parameter.get('label_path', 'label.npy')

    def begin(self):

        epochs = self.epochs
        using_adj = self.parameter.get('adj_path')
        using_cli_demo = self.parameter.get('cli_demo_path')
        
        for archive in self.all_archive:
            hbo_fold_path = default_hb_fold_path + archive
            fnirs_data_path = preprocessed_hb_fold_path + \
                model_name if model_name in self.config.MODELS_NEED_PREPROCESS_DATA else hbo_fold_path
            for classifier_name in self.all_classifiers:
                # Read data and split into training+validation and testing set with a ratio of 9:1
                # case - not using adj
                # case using adj include GNN, GNN-Transformer, ....
                if using_adj:
                    if using_cli_demo:
                        data, label, adj, cli_demo = simply_read_data_fnirs(
                            fnirs_data_path, self.label_path, self.hb_path, self.adj_path, cli_demo_path=using_cli_demo)
                    else:
                        data, label, adj = simply_read_data_fnirs(
                        fnirs_data_path, self.label_path, self.hb_path, self.adj_path)
                else:
                    data, label = simply_read_data_fnirs(
                        fnirs_data_path, self.label_path, self.hb_path, None)

                
                num_of_k_fold = config.SPECIFY_FOLD
                self.params = params = info['parameter']
                msg = info['message'] # + config_name #get_params_info(params)
                for k in range(num_of_k_fold):
                    if using_adj:
                        X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test = stratified_k_fold_cross_validation_with_holdout(
                            data, label, k, num_of_k_fold, adj, seed=current_time, hold_out_div=config.HOLD_OUT_DIV)
                        if using_cli_demo:
                            X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test, cli_demo_train, cli_demo_val, cli_demo_test = stratified_k_fold_cross_validation_with_holdout_with_cli_demo(
                            data, label, cli_demo, k, num_of_k_fold, adj, seed=current_time, hold_out_div=config.HOLD_OUT_DIV)
                    else:
                        X_train, Y_train, X_val, Y_val, X_test, Y_test = stratified_k_fold_cross_validation_with_holdout(
                            data, label, k, num_of_k_fold, seed=current_time, hold_out_div=config.HOLD_OUT_DIV)

                    # reading data will set the random.seed
                    # random.seed(current_time)
                    # np.random.seed(current_time)
                    # tf.random.set_seed(current_time)
                    # print(f'X_train shape: {X_train.shape}'*99)
                    # Augment data
                    if config.AUGMENT_RATIO != 0: X_train, Y_train = augment_data(X_train, Y_train, ratio=config.AUGMENT_RATIO, min_delete_ch=config.MIN_DELETE_CHANNEL, max_delete_ch=config.MAX_DELETE_CHANNEL)
                    print(f'X_train - {X_train.shape}, X_val - {X_val.shape}, X_test - {X_test.shape}')
                             
                    # msg = info['message'] + f"d_model_{params['d_model']}_batch_size_{params['batch_size']}_n_layers_{params['n_layers']}"
                    output_directory = os.getcwd() + '/results/' + classifier_name + '/' + \
                    archive + \
                    f'/{msg}/' + f"SKF_holdout/stratified_nested_{num_of_k_fold}_CV_fold-{str(k)}" + '/'

                    print(f'output_directory -> {output_directory}')
                    create_directory(output_directory)

                    if model_name in ['chao_cfnn', 'zhu_xgboost', 'decision_tree']:
                        input_shape = [self.batch_size,
                                    X_train.shape[1]]
                    elif model_name in ['mvg_transformer', 'mgn_transformer', 'mgm_transformer']:
                        input_shape = [self.batch_size,
                                    X_train.shape[1],
                                    X_train.shape[2],
                                    adj_train.shape[-1]]
                    else:
                        input_shape = [self.batch_size] + list(X_train.shape[1:])


                    callbacks = []
                    if using_wandb:
                        callbacks.append(WandbCallback(save_model=False))

                    model = create_classifier(
                        classifier_name, output_directory, callbacks, input_shape, epochs, info, self.sweep_config)

                    if using_adj:
                        if using_cli_demo:
                            model.fit(X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test, cli_demo_train, cli_demo_val, cli_demo_test)
                        else:
                            model.fit(X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test)

                    else:
                        model.fit(X_train, Y_train, X_val,
                                Y_val, X_test, Y_test)
                    del model
                    del X_train, Y_train, X_val, Y_val, X_test, Y_test
                    if using_adj:
                        del adj_train, adj_val, adj_test
                    # clear the memory
                    tf.keras.backend.clear_session()
                    gc.collect()

                        # if wandb is activated, then we only calculate the k=0 fold cross validation for 25 times
                    if using_wandb:
                        break


model_names = ['transformer', 'gnn_transformer',
               'dim_transformer', 'dgi_transformer', 'yu_gnn']
if __name__ == '__main__':
    arg = sys.argv
    model_name = arg[1]
    config_file_name = 'configs.' + arg[3]
    config_name = arg[3]
    current_time = int(arg[4])
    set_seed(current_time)
    print('config_file_name', config_file_name)
    config = importlib.import_module(config_file_name)
    preprocessed_hb_fold_path = config.PREPROCESSED_HB_FOLD_PATH
    default_hb_fold_path = config.DEFAULT_HB_FOLD_PATH
    info = {'current_time_seed': current_time,
            'message': arg[2],
            'parameter': config.PARAMETER[model_name],
            'monitor_metric': config.MONITOR_METRIC
            }
    
    print('You are using model: {}'.format(model_name))
    using_wandb = config.IS_USING_WANDB

    model = TrainModel(model_name, config=config)
    model.begin()
