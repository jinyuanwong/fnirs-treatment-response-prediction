
"""df_metrics.csv 里面sensitivity计算可能有问题注意修改"""
"""
This is used to stage low+mid(8-23) HAMD and High(>=24) HAMD score MDD subjects with label 0 and 1 

Normalization Method: Layer Normalization (Single Sample Normalization)

Data Augmentation: None

"""


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
import config
import gc
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

# hbo_fold_path = './allData/Output_npy/twoDoctor/nor-all-hbo-hc-mdd'
num_of_k_fold = 10
# /home/jy/Documents/JinyuanWang_pythonCode/results/wang_alex/HbO-All-HC-MDD


def split_k_fold_cross_validation(data, label, k, num_of_k_fold, adj=None):
    total_number = data.shape[0]
    one_fold_number = total_number//num_of_k_fold
    X_val = data[k*one_fold_number:(k+1)*one_fold_number]
    Y_val = label[k*one_fold_number:(k+1)*one_fold_number]
    X_train = np.concatenate(
        (data[0:k*one_fold_number], data[(k+1)*one_fold_number:]))
    Y_train = np.concatenate(
        (label[0:k*one_fold_number], label[(k+1)*one_fold_number:]))

    if adj is None:
        return X_train, Y_train, X_val, Y_val
    else:
        adj_val = adj[k*one_fold_number:(k+1)*one_fold_number]
        adj_train = np.concatenate(
            (adj[0:k*one_fold_number], adj[(k+1)*one_fold_number:]))
        return X_train, Y_train, X_val, Y_val, adj_train, adj_val


class TrainModel():
    def __init__(self, model_name, sweep_config=None):
        self.batch_size = 8
        self.epochs = config.MAX_EPOCHS
        # ['nor-all-hbo-hc-mdd']  # 'HbO-All-Three'
        self.all_archive = config.INPUT_HB_TYPE
        self.all_classifiers = [model_name]
        self.repeat_count_all = 1
        self.sweep_config = sweep_config
        self.parameter = config.PARAMETER[model_name]
        self.hb_path = self.parameter.get('hb_path')
        self.adj_path = self.parameter.get('adj_path')
    def begin(self):
        
        epochs = self.epochs
        using_adj = self.parameter.get('adj_path')

        for archive in self.all_archive:
            hbo_fold_path = default_hb_fold_path + archive
            fnirs_data_path = preprocessed_hb_fold_path + \
                model_name if model_name in config.MODELS_NEED_PREPROCESS_DATA else hbo_fold_path
            for classifier_name in self.all_classifiers:

                # Read data and split into training+validation and testing set with a ratio of 9:1
                # case - not using adj
                # case using adj include GNN, GNN-Transformer, ....
                if using_adj:
                    X_train_val, X_test, Y_train_val, Y_test, adj_train_val, adj_test = read_data_fnirs(
                        fnirs_data_path, model_name, self.hb_path, self.adj_path)
                else:
                    X_train_val, X_test, Y_train_val, Y_test = read_data_fnirs(
                        fnirs_data_path, model_name, self.hb_path, None)
                for k in range(num_of_k_fold):

                    if using_adj:
                        X_train, Y_train, X_val, Y_val, adj_train, adj_val = split_k_fold_cross_validation(
                            X_train_val, Y_train_val, k, num_of_k_fold, adj_train_val)
                    else:
                        X_train, Y_train, X_val, Y_val = split_k_fold_cross_validation(
                            X_train_val, Y_train_val, k, num_of_k_fold)

                    output_directory = os.getcwd() + '/results/' + classifier_name + '/' + \
                        archive + \
                        '/' + 'k-fold-' + str(k) + '/'
                    create_directory(output_directory)

                    checkpoint_path = output_directory + '/checkpoint'

                    def learning_rate_schedule(epoch, learning_rate):
                        return learning_rate

                    lr_monitor = tf.keras.callbacks.LearningRateScheduler(
                        learning_rate_schedule)

                    if model_name in ['chao_cfnn', 'zhu_xgboost']:
                        input_shape = [self.batch_size,
                                       X_train_val.shape[1]]
                    elif model_name in ['comb_cnn', 'cnn_transformer']:
                        input_shape = [self.batch_size,
                                       X_train_val.shape[1],
                                       X_train_val.shape[2],
                                       1]
                    elif model_name in ['mvg_transformer', 'mgn_transformer', 'mgm_transformer']:
                        input_shape = [self.batch_size,
                                       X_train_val.shape[1],
                                       X_train_val.shape[2],
                                       adj_train_val.shape[-1]]
                    else:
                        input_shape = [self.batch_size,
                                       X_train_val.shape[1],
                                       X_train_val.shape[2]]
                    
                    for repeat_count in range(self.repeat_count_all):

                        model_checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                                           monitor= 'val_' + config.MONITOR_METRIC,
                                                           mode = 'max',
                                                           save_weights_only=True,
                                                           save_best_only=True)

                        callbacks = [model_checkpoint,
                                     lr_monitor]
                        if using_wandb:
                            callbacks.append(WandbCallback(save_model=False))

                        tf.keras.backend.clear_session()
                        print(
                            f'Current / Total repeat count: {repeat_count} / {self.repeat_count_all}')

                        model = self.create_classifier(
                            classifier_name, output_directory, callbacks, input_shape, epochs, self.sweep_config)

                        if using_adj:
                            model.fit(X_train, Y_train, X_val, Y_val,
                                      X_test, Y_test, adj_train, adj_val, adj_test)
                        else:
                            model.fit(X_train, Y_train, X_val,
                                      Y_val, X_test, Y_test)

                        del model
                        del X_train, Y_train, X_val, Y_val
                        # clear the memory
                        gc.collect()


                        # if wandb is activated, then we only calculate the k=0 fold cross validation for 25 times
                    if using_wandb:
                        break
                    # for obj in gc.get_objects():
                    #     print(type(obj), repr(obj))
                        
                        

    def create_classifier(self, classifier_name, output_directory, callbacks, input_shape, epochs, sweep_config=None):
        if classifier_name == 'cnn_transformer':  # Time-CNN
            from classifiers import cnn_transformer
            return cnn_transformer.Classifier_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        if classifier_name == 'transformer':  # Time-CNN
            from classifiers import transformer
            return transformer.Classifier_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        if classifier_name == 'gnn':  # Time-CNN
            from classifiers import gnn
            return gnn.Classifier_GNN(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        if classifier_name == 'gnn_transformer':  # Time-CNN
            from classifiers import gnn_transformer
            return gnn_transformer.Classifier_GNN_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        if classifier_name == 'gin_transformer':  # Time-CNN
            from classifiers import gin_transformer
            return gin_transformer.Classifier_GIN_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        if classifier_name == 'rggcnn_transformer':  # Time-CNN
            from classifiers import rggcnn_transformer
            return rggcnn_transformer.Classifier_RGGCNN_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        if classifier_name == 'mvg_transformer':  # Time-CNN
            from classifiers import mvg_transformer
            return mvg_transformer.Classifier_MVG_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        if classifier_name == 'mgn_transformer':  # Time-CNN
            from classifiers import mvg_transformer
            return mvg_transformer.Classifier_MVG_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        if classifier_name == 'mgm_transformer':  # Time-CNN
            from classifiers import mgm_transformer
            return mgm_transformer.Classifier_MGM_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        if classifier_name == 'graphsage_transformer':  # Time-CNN
            from classifiers import graphsage_transformer
            return graphsage_transformer.Classifier_GraphSAGE_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        if classifier_name == 'dgi_transformer':  # Time-CNN
            from classifiers import dgi_transformer
            return dgi_transformer.Classifier_Transformer(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        if classifier_name == 'yu_gnn':
            from classifiers import yu_gnn
            return yu_gnn.Classifier_GCN(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        if classifier_name == 'wang_alex':
            from classifiers import wang_alex
            return wang_alex.Classifier_AlexNet(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        if classifier_name == 'comb_cnn':  # Time-CNN
            from classifiers import comb_cnn
            return comb_cnn.Classifier_CNN(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        if classifier_name == 'chao_cfnn':  # Time-CNN
            from classifiers import chao_cfnn
            return chao_cfnn.Classifier_CFNN(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        if classifier_name == 'zhu_xgboost':  # Time-CNN
            from classifiers import zhu_xgboost
            return zhu_xgboost.Classifier_XGBoost(output_directory, callbacks, input_shape, epochs, sweep_config, info)
        else:
            raise Exception('Your error message here')


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
        'gnn_layers':{
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
    do_individual_normalize = True
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
        model = TrainModel(model_name)
        model.begin()
