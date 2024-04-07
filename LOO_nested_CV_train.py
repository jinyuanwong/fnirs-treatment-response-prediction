

# 使用当前时间作为随机种子
from wandb.keras import WandbCallback
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
import wandb
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

    def begin(self):

        epochs = self.epochs
        using_adj = self.parameter.get('adj_path')

        for archive in self.all_archive:
            hbo_fold_path = default_hb_fold_path + archive
            fnirs_data_path = preprocessed_hb_fold_path + \
                model_name if model_name in self.config.MODELS_NEED_PREPROCESS_DATA else hbo_fold_path
            for classifier_name in self.all_classifiers:
                # Read data and split into training+validation and testing set with a ratio of 9:1
                # case - not using adj
                # case using adj include GNN, GNN-Transformer, ....
                if using_adj:
                    data, label, adj = simply_read_data_fnirs(
                        fnirs_data_path, model_name, self.hb_path, self.adj_path)
                else:
                    data, label = simply_read_data_fnirs(
                        fnirs_data_path, model_name, self.hb_path, None)
                if SPECIFY_FOLD: 
                    num_of_k_fold = SPECIFY_FOLD
                else:
                    label_not_one_hot = np.argmax(label, axis=1)
                    num_of_k_fold = 3 # I think 3 will be good as pre-treatment data has 15 positive samples and posttreatment has around 12 positive smaples

                params = info['parameter']
                msg = info['message'] + get_params_info(params)
                loo_array = get_sorted_loo_array(model_name, msg, data.shape[0], DATASET=archive, K_FOLD=num_of_k_fold)
                print('loo_array', loo_array)
                for current_loo in loo_array:#  loo_array:# range(loo_start_from, data.shape[0]): #
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
                        print(f"total sample size is {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}")
                        
                        # msg = info['message'] + f"d_model_{params['d_model']}_batch_size_{params['batch_size']}_n_layers_{params['n_layers']}"
                        output_directory = os.getcwd() + '/results/' + classifier_name + '/' + \
                        archive + \
                        f'/{msg}/' + f"LOO_nested_CV/LOO_{current_loo}/stratified_nested_{num_of_k_fold}_CV_fold-{str(k)}" + '/'

                        print(f'output_directory -> {output_directory}')
                        create_directory(output_directory)

                        checkpoint_path = output_directory + 'checkpoint'

                        def learning_rate_schedule(epoch, learning_rate):
                            return learning_rate

                        lr_monitor = tf.keras.callbacks.LearningRateScheduler(
                            learning_rate_schedule)

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


                        model_checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                                        monitor='val_' + self.config.MONITOR_METRIC,
                                                        mode='max',
                                                        save_weights_only=True,
                                                        save_best_only=True)

                        callbacks = [model_checkpoint,
                                    lr_monitor]
                        if using_wandb:
                            callbacks.append(WandbCallback(save_model=False))

                        model = create_classifier(
                            classifier_name, output_directory, callbacks, input_shape, epochs, info, self.sweep_config)

                        if using_adj:
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


def train_model():
    wandb.init()  # mode='disabled'
    config = wandb.config
    if config is None:
        raise ValueError("config is None")
    print(f"Config: {config}")
    model = TrainModel(model_name, config)
    if model_name == 'dgi_transformer':
        model.begin()
    elif model_name == 'gnn_transformer':
        model.begin()
    elif model_name == 'yu_gnn':
        model.begin()
    elif model_name == 'wang_alex':
        model.begin()
    elif model_name == 'cnn_transformer':
        model.begin()
    elif model_name == 'chao_cfnn':
        model.begin()
    elif model_name == 'zhu_xgboost':
        model.begin()
    elif model_name == 'mvg_transformer':
        model.begin()
    wandb.finish()

# for yu_gnn


def using_sweep_for_yu_gnn():

    sweep_config = {
        'method': 'random'
    }
    metric = {
        'name': 'val_loss',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric
    parameters_dict = {
        'activation': {
            'values': ['relu', 'prelu']
        },
        'lr': {
            'values': [0.5, 0.1, 1e-2]
        },
        'd_model_1': {
            'values': [10, 20]
        },
        'batch_size': {
            'values': [128, 256]
        }
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="yu_gnn")

    wandb.agent(sweep_id, function=train_model, count=10)

# for wang_alex


def using_sweep_for_wang_alex():

    sweep_config = {
        'method': 'random'
    }
    metric = {
        'name': 'val_loss',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric
    parameters_dict = {
        'lr': {
            'values': [0.5, 0.1, 1e-2]
        },
        'dropout2': {
            'values': [0.7, 0.5, 0.3, 0.1]
        },
        'dense2': {
            'values': [256, 128, 64]
        },
        'batch_size': {
            'values': [64, 128, 256]
        }
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="wang_alex")

    wandb.agent(sweep_id, function=train_model, count=25)

# for chao_cfnn


def using_sweep_for_chao_cfnn():

    sweep_config = {
        'method': 'random'
    }
    metric = {
        'name': 'val_loss',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric
    parameters_dict = {
        'lr': {
            'values': [0.5, 0.1, 1e-2]
        },
        'activation': {
            'values': ['relu', 'prelu', 'sigmoid']
        },
        'dense0': {
            'values': [256, 128, 64]
        },
        'batch_size': {
            'values': [64, 128, 256]
        }
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="chao_cfnn")

    wandb.agent(sweep_id, function=train_model, count=25)

# for zhu_xgboost


def using_sweep_for_zhu_xgboost():
    pass


def using_sweep_for_gnn_transformer():

    sweep_config = {
        'method': 'random'
    }
    metric = {
        'name': 'val_loss',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric
    parameters_dict = {
        'FFN_units': {
            'values': [64, 128, 256, 512]
        },
        'gnn_layers': {
            'values': [1, 2, 3, 4, 5]
        },
        'n_layers': {
            'values': [4, 8, 12]
        },
        'lr_factor': {
            'values': [0.001, 0.01, 0.1, 10]
        }
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="gnn_transformer")
    wandb.agent(sweep_id, function=train_model, count=30)


def using_sweep_for_mvg_transformer():

    sweep_config = {
        'method': 'random'
    }
    metric = {
        'name': 'val_loss',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric
    parameters_dict = {
        'FFN_units': {
            'values': [64, 256]
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'n_heads': {
            'values': [1, 2, 4]
        },
        'activation': {
            'values': ['relu', 'sigmoid']
        },
        'mvg_layer': {
            'values': [2, 3, 4, 5]
        }
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="mvg_transformer")
    wandb.agent(sweep_id, function=train_model, count=25)

# for gnn_transformer


def using_sweep_for_dgi_transformer():

    sweep_config = {
        'method': 'random'
    }
    metric = {
        'name': 'val_loss',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric
    parameters_dict = {
        'n_layers': {
            'values': [4, 8, 12]
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'lr_rate': {
            'values': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="dgi_transformer")

    wandb.agent(sweep_id, function=train_model, count=20)


model_names = ['transformer', 'gnn_transformer',
               'dim_transformer', 'dgi_transformer', 'yu_gnn']
if __name__ == '__main__':
    arg = sys.argv
    model_name = arg[1]
    config_file_name = 'configs.' + arg[3]
    config = importlib.import_module(config_file_name)
    preprocessed_hb_fold_path = config.PREPROCESSED_HB_FOLD_PATH
    default_hb_fold_path = config.DEFAULT_HB_FOLD_PATH
    SPECIFY_FOLD = config.SPECIFY_FOLD
    info = {'current_time_seed': current_time,
            'message': arg[2],
            'parameter': config.PARAMETER[model_name],
            'monitor_metric': config.MONITOR_METRIC
            }
    
    print('You are using model: {}'.format(model_name))
    using_wandb = config.IS_USING_WANDB
    if model_name == 'zhu_xgboost':
        using_wandb = False
    if using_wandb == True:
        wandb.login()
        if model_name == 'dgi_transformer':
            using_sweep_for_dgi_transformer()
        elif model_name == 'gnn_transformer':
            using_sweep_for_gnn_transformer()
        elif model_name == 'yu_gnn':
            using_sweep_for_yu_gnn()
        elif model_name == 'wang_alex':
            using_sweep_for_wang_alex()
        elif model_name == 'chao_cfnn':
            using_sweep_for_chao_cfnn()
        elif model_name == 'zhu_xgboost':
            # using_sweep_for_zhu_xgboost()
            raise NotImplementedError(
                'Currently sweep for zhu_xgboost is not implemented yet.')
        elif model_name == 'mvg_transformer':
            using_sweep_for_mvg_transformer()
            raise NotImplementedError(
                'Currently sweep for mvg_transformer is not implemented yet.')
        elif model_name == 'graphsage_transformer':
            raise NotImplementedError(
                'Currently sweep for mvg_transformer is not implemented yet.')
    else:
        model = TrainModel(model_name, config=config)
        model.begin()
