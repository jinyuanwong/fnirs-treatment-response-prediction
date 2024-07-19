


# 使用当前时间作为随机种子
# from wandb.keras import WandbCallback
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



class TrainModel():
    def __init__(self, model_name, config, sweep_config=None):
        self.config = config
        self.epochs = config.MAX_EPOCHS
        self.batch_size = config.PARAMETER[model_name].get('batch_size', 8)
        # ['nor-all-hbo-hc-mdd']  # 'HbO-All-Three'
        self.all_archive = config.INPUT_HB_TYPE[0] # only consider using the first dataset of INPUT_HY_TYPE
        self.parameter = config.PARAMETER[model_name]
        self.model_mame = model_name
        self.sweep_config = sweep_config
        self.hb_path = self.parameter.get('hb_path')
        self.adj_path = self.parameter.get('adj_path')
        self.cli_demo_path = self.parameter.get('cli_demo_path')
        self.label_path = self.parameter.get('label_path', 'label.npy')
        
    def read_data_label(self, fnirs_data_path):
        if self.adj_path:
            if self.cli_demo_path:
                data, label, adj, cli_demo = simply_read_data_fnirs(
                    fnirs_data_path, self.label_path, self.hb_path, self.adj_path, cli_demo_path=self.cli_demo_path)
                self.cli_demo = cli_demo
            else:
                data, label, adj = simply_read_data_fnirs(
                fnirs_data_path, self.label_path, self.hb_path, self.adj_path)
            self.adj = adj
        else:
            data, label = simply_read_data_fnirs(
                fnirs_data_path, self.label_path, self.hb_path, None)
        self.data, self.label = data, label
        
    def nested_cross_validation(self, inner_k, total_inner_k, outer_k, total_outer_k, seed, MSG):
        msg = MSG
        classifier_name = self.model_mame
        archive = self.all_archive
        
        if self.adj_path:
            X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test = nested_cross_validation_split(
                self.data, self.label, inner_k, total_inner_k, outer_k, total_outer_k, self.adj, seed=seed)
            if self.cli_demo_path:
                X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test, cli_demo_train, cli_demo_val, cli_demo_test = nested_cross_validation_split(
                self.data, self.label, self.cli_demo, inner_k, total_inner_k, outer_k, total_outer_k, self.adj, seed=seed)
                self.cli_demo_train, self.cli_demo_val, self.cli_demo_test = cli_demo_train, cli_demo_val, cli_demo_test
            self.adj_train, self.adj_val, self.adj_test = adj_train, adj_val, adj_test
        else:
            X_train, Y_train, X_val, Y_val, X_test, Y_test = nested_cross_validation_split(
                self.data, self.label, inner_k, total_inner_k, outer_k, total_outer_k, seed=seed)
        

        if config.AUGMENT_RATIO != 0: X_train, Y_train = augment_data(X_train, Y_train, ratio=config.AUGMENT_RATIO, min_delete_ch=config.MIN_DELETE_CHANNEL, max_delete_ch=config.MAX_DELETE_CHANNEL)
        print(f'X_train - {X_train.shape}, X_val - {X_val.shape}, X_test - {X_test.shape}')
                    
        output_directory = os.getcwd() + '/results/' + classifier_name + '/' + \
        archive + \
        f'/{msg}/' + f"nested_cross_validation_outer_{total_outer_k}_inner_{total_inner_k}/outer_{outer_k}_inner_{inner_k}" + '/'

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

        
        self.output_directory = output_directory
        self.callbacks = callbacks
        self.input_shape = input_shape  
        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = X_train, Y_train, X_val, Y_val, X_test, Y_test
    
    """
    this will set the train, val, test data for each fold
    """
    def stratified_k_fold_cross_validation_with_holdout(self, k, num_of_k_fold, seed, hold_out_div, MSG):
        msg = MSG
        classifier_name = self.model_mame
        archive = self.all_archive
        
        if self.adj_path:
            X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test = stratified_k_fold_cross_validation_with_holdout(
                self.data, self.label, k, num_of_k_fold, self.adj, seed=seed, hold_out_div=hold_out_div)
            if self.cli_demo_path:
                X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test, cli_demo_train, cli_demo_val, cli_demo_test = stratified_k_fold_cross_validation_with_holdout_with_cli_demo(
                self.data, self.label, self.cli_demo, k, num_of_k_fold, self.adj, seed=seed, hold_out_div=hold_out_div)
                self.cli_demo_train, self.cli_demo_val, self.cli_demo_test = cli_demo_train, cli_demo_val, cli_demo_test
            self.adj_train, self.adj_val, self.adj_test = adj_train, adj_val, adj_test
        else:
            X_train, Y_train, X_val, Y_val, X_test, Y_test = stratified_k_fold_cross_validation_with_holdout(
                self.data, self.label, k, num_of_k_fold, seed=seed, hold_out_div=hold_out_div)
        

        if config.AUGMENT_RATIO != 0: X_train, Y_train = augment_data(X_train, Y_train, ratio=config.AUGMENT_RATIO, min_delete_ch=config.MIN_DELETE_CHANNEL, max_delete_ch=config.MAX_DELETE_CHANNEL)
        print(f'X_train - {X_train.shape}, X_val - {X_val.shape}, X_test - {X_test.shape}')
                    
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

        
        self.output_directory = output_directory
        self.callbacks = callbacks
        self.input_shape = input_shape  
        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test = X_train, Y_train, X_val, Y_val, X_test, Y_test
        
        
        
    def build_classifier(self, info):
        self.model = create_classifier(
            self.model_mame, self.output_directory, self.callbacks, self.input_shape, self.epochs, info, self.sweep_config)

    def train(self, ):
        if self.adj_path:
            if self.cli_demo_path:
                self.model.fit(self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test, self.adj_train, self.adj_val, self.adj_test, self.cli_demo_train, self.cli_demo_val, self.cli_demo_test)
            else:
                self.model.fit(self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test, self.adj_train, self.adj_val, self.adj_test)
        else:
            self.model.fit(self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test)
        del self.model
        del self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test
        
    def split_data_and_build_model_to_train(self, inner_k, outer_k, info, config, current_time, MSG):
    
        print(f'using nested_cross_validation inner_fold: {inner_k} outer_fold: {outer_k}...')
        classifier.nested_cross_validation(inner_k, config.SPECIFY_FOLD, outer_k, config.OUTER_FOLD, current_time, MSG)
        print('Successfully nested_cross_validation...')
        print('Building classifier...')
        classifier.build_classifier(info)
        print('Successfully built classifier...')
        print('Training classifier...')
        classifier.train()

model_names = ['transformer', 'gnn_transformer',
               'dim_transformer', 'dgi_transformer', 'yu_gnn']
if __name__ == '__main__':
    arg = sys.argv
    model_name = arg[1]
    config_file_name = 'configs.' + arg[3]
    config_name = arg[3]
    current_time = int(arg[4])
    set_seed(current_time)


    config = importlib.import_module(config_file_name)
    default_hb_fold_path = config.DEFAULT_HB_FOLD_PATH
    print('config_file_name', config_file_name)
    tf.keras.backend.set_floatx(config.DATA_PRECISION)
    # save current file 
    config.PARAMETER[model_name]['config_file_path'].append(os.path.abspath(__file__))
    
    MSG = arg[2] + arg[3] # message = arg[2]
    
    info = {'current_time_seed': current_time,
            'message': MSG,
            'parameter': config.PARAMETER[model_name],
            'monitor_metric': config.MONITOR_METRIC
            }
    
    preprocessed_hb_fold_path = config.PREPROCESSED_HB_FOLD_PATH
    hbo_fold_path = default_hb_fold_path + config.INPUT_HB_TYPE[0]
    fnirs_data_path = preprocessed_hb_fold_path + model_name if model_name in config.MODELS_NEED_PREPROCESS_DATA else hbo_fold_path
    
    print('You are using model: {}'.format(model_name))
    
    classifier = TrainModel(model_name, config=config)

    classifier.read_data_label(fnirs_data_path)
    for outer_k in range(config.OUTER_FOLD):
        for inner_k in range(config.SPECIFY_FOLD):
            classifier.split_data_and_build_model_to_train(inner_k, outer_k, info, config, current_time, MSG)
    # for k in range(config.SPECIFY_FOLD):
    #     print(f'using stratified_k_fold_cross_validation_with_holdout...{k}')
    #     classifier.stratified_k_fold_cross_validation_with_holdout(k, config.SPECIFY_FOLD, current_time, config.HOLD_OUT_DIV, MSG)
    #     print('Successfully stratified_k_fold_cross_validation_with_holdout...')
    #     print('Building classifier...')
    #     classifier.build_classifier(info)
    #     print('Successfully built classifier...')
    #     print('Training classifier...')
    #     classifier.train()
    
    
    # using_wandb = config.IS_USING_WANDB

    # model = TrainModel(model_name, config=config)
    # model.begin()
