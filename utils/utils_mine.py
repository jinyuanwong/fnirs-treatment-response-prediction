
import os
import sys
import scipy.sparse as sp
from tensorflow.keras.models import save_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa

from sklearn.metrics import recall_score
from tensorflow.keras.metrics import Recall
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_auc_score
# 保存日志
import logging

import random
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut

import shutil

import math 
def get_params_info(params):
    return_string = ''
    for key, value in params.items():
        if key not in ['hb_path', 'adj_path', 'cli_demo_path', 'cli_demo_shape', 'cli_dense_units']:
            if return_string != '':
                return_string += '_'
            return_string += f"{key}_{value}"
    return return_string

def update_config_file(key, new_value):
    # Path to your config.py file
    config_file_path = 'config.py'

    # Read the current contents of the file
    with open(config_file_path, 'r') as file:
        lines = file.readlines()

    # Modify the desired value
    for i, line in enumerate(lines):
        if line.startswith(key):
            lines[i] = f"{key} = {repr(new_value)}\n"

    # Write the changes back to the file
    with open(config_file_path, 'w') as file:
        file.writelines(lines)
        
        
def get_specificity(y_true, y_pred):

    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 明确指定labels参数
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 现在cm是一个2x2矩阵，即使数据只包含一个类别
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn+fp)


def predict(model, x_test, y_test, return_df_metrics=True):
    y_pred = model.predict(x_test)
    if return_df_metrics:
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        df_metrics = accuracy_score(y_test, y_pred)
        return df_metrics
    else:
        return y_pred


def z_norm(data, normalization_method):

    if normalization_method == 0:
        # all sample normalization
        print('you are using all sample nor')
        td_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
        print(td_data.shape)
        scaler = preprocessing.StandardScaler().fit(td_data)

        td_data = scaler.transform(td_data)
        new_data = td_data.reshape(data.shape[0], data.shape[1], data.shape[2])
        return new_data
    else:
        print('you are using every sample nor')
        new_data = np.empty_like(data)
        for i in range(data.shape[0]):
            # Extract the 2D data for the current sample (assuming 1 in the last dimension)
            sample = data[i, :, :]
            scaler = preprocessing.StandardScaler().fit(sample)
            normalized_sample = scaler.transform(sample)
            new_data[i, :, :] = normalized_sample
        return new_data


def onehotEncode(x):
    x = x.astype(int)
    t = np.zeros((x.size, x.max()+1))
    t[np.arange(x.size), x] = 1
    return t.astype(int)


def shuffle(x_data, y_data):
    length = np.array(range(x_data.shape[0]))
    np.random.shuffle(length)
    new_x_data = np.zeros(x_data.shape)
    new_y_data = np.zeros(y_data.shape)
    for i in range(x_data.shape[0]):
        new_x_data[i] = x_data[length[i]]
        new_y_data[i] = y_data[length[i]]
    return new_x_data, new_y_data


def plot_epochs_metric(hist, file_name, metric='loss') -> object:
    data = hist.history
    fig = plt.figure(figsize=(16, 9))
    acc = data['accuracy']
    loss = data['loss']
    v_acc = data['val_accuracy']
    v_loss = data['val_loss']
    ax1 = fig.add_subplot(2, 2, 1)
    # ax1.set_aspect('equal', adjustable='box')
    ax1.plot(v_acc)
    ax1.set_title('Validation Accuracy')
    plt.xlabel('epoch', fontsize='large')
    ax2 = fig.add_subplot(2, 2, 2)
    # ax2.set_aspect('equal', adjustable='box')
    ax2.plot(v_loss)
    ax2.set_title('Validation Loss')
    plt.xlabel('epoch', fontsize='large')
    ax3 = fig.add_subplot(2, 2, 3)
    # ax3.set_aspect('equal', adjustable='box')
    ax3.plot(loss)
    ax3.set_title('Loss')
    plt.xlabel('epoch', fontsize='large')
    ax4 = fig.add_subplot(2, 2, 4)
    # ax4.set_aspect('equal', adjustable='box')
    ax4.plot(acc)
    ax4.set_title('Accuracy')
    plt.xlabel('epoch', fontsize='large')
    fig.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def assignTrainAndTest(x_data, y_data):
    length = np.load(os.getcwd()+'/results/all_le_shuffle.log.npy')
    all_accuracy = np.load(os.getcwd()+'/results/all_ac_shuffle.log.npy')
    length = length[np.argmax(np.max(all_accuracy, axis=1))]
    # all_length = np.concatenate((all_length,length.reshape(1,-1)))
    new_x_data = np.zeros(x_data.shape)
    new_y_data = np.zeros(y_data.shape)
    for i in range(x_data.shape[0]):
        new_x_data[i] = x_data[length[i]]
        new_y_data[i] = y_data[length[i]]
    all_X = new_x_data
    all_Y = new_y_data
    length_test = int(all_Y.shape[0]*0.2)
    X_train = all_X[0:-length_test]
    Y_train = all_Y[0:-length_test]
    X_test = all_X[-length_test:]
    Y_test = all_Y[-length_test:]
    return X_train, Y_train, X_test, Y_test


class Methods():
    def __init__(self):
        pass

    def Shuffle(self, _data, _label):
        print(len(_label))
        array = list(range(len(_label)))
        random.shuffle(array)
        newData, newLabel = np.empty(
            shape=_data.shape), np.empty(shape=_label.shape)
        for i in range(len(array)):
            newData[i, :, :] = _data[array[i], :, :]
            newLabel[i] = _label[array[i]]
        return newData, newLabel


def read_data_split(file_name, normalization_method=1, random_state=random.randint(0, 1000)):

    data = np.load(file_name + '/data.npy')  # correct_channel_
    data = z_norm(data, normalization_method)
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))
    label = np.load(file_name + '/label.npy')
    label = onehotEncode(label.astype(int))
    X_train, X_test, Y_train, Y_test = train_test_split(
        data, label, test_size=0.25, random_state=random_state)

    return X_train, X_test, Y_train, Y_test


def generate_fnirs_adj():
    matrix = sp.csr_matrix((52, 52), dtype=int)
    for i in range(10):
        if i > 0 and i < 9:
            matrix[i, i-1] = 1
            matrix[i, i+1] = 1
        elif i == 0:
            matrix[i, i+1] = 1
        else:
            matrix[i, i-1] = 1
        matrix[i, i+10] = 1
        matrix[i, i+11] = 1

    for i in range(10, 21):
        if i > 10 and i < 20:
            matrix[i, i-11] = 1
            matrix[i, i-10] = 1
            matrix[i, i-1] = 1
            matrix[i, i+1] = 1
            matrix[i, i+10] = 1
            matrix[i, i+11] = 1
        if i == 10:
            matrix[i, i-10] = 1
            matrix[i, i+1] = 1
            matrix[i, i+11] = 1
        if i == 20:
            matrix[i, i-11] = 1
            matrix[i, i-1] = 1
            matrix[i, i+10] = 1

    for i in range(21, 31):
        matrix[i, i-11] = 1
        matrix[i, i-10] = 1
        if i > 21 and i < 30:
            matrix[i, i-1] = 1
            matrix[i, i+1] = 1
        if i == 21:
            matrix[i, i+1] = 1

        if i == 30:
            matrix[i, i-1] = 1
        matrix[i, i+10] = 1
        matrix[i, i+11] = 1

    begin = 31
    end = 42

    for i in range(begin, end):
        if i > begin and i < end-1:
            matrix[i, i-11] = 1
            matrix[i, i-10] = 1
            matrix[i, i-1] = 1
            matrix[i, i+1] = 1
            matrix[i, i+10] = 1
            matrix[i, i+11] = 1
        if i == begin:
            matrix[i, i-10] = 1
            matrix[i, i+1] = 1
            matrix[i, i+11] = 1
        if i == end-1:
            matrix[i, i-11] = 1
            matrix[i, i-1] = 1
            matrix[i, i+10] = 1

    begin = 42
    end = 52

    for i in range(begin, end):
        if i > begin and i < end-1:
            matrix[i, i-11] = 1
            matrix[i, i-10] = 1
            matrix[i, i-1] = 1
            matrix[i, i+1] = 1
        if i == begin:
            matrix[i, i-11] = 1
            matrix[i, i-10] = 1
            matrix[i, i+1] = 1
        if i == end-1:
            matrix[i, i-11] = 1
            matrix[i, i-10] = 1
            matrix[i, i-1] = 1
    return matrix

def generate_fnirs_adj_tf():
    sp_matrix = generate_fnirs_adj()
    matrix_arr = sp_matrix.toarray()
    tf_matrix = tf.convert_to_tensor(matrix_arr, dtype=tf.keras.backend.floatx())
    return tf_matrix


def generate_adj_for_mvg(connect_file_path='./allData/Output_npy/twoDoctor/HbO-All-HC-MDD/multiview_adj_matrix5.npy'):
    connectivity = np.load(connect_file_path)
    return connectivity


def save_data_to_file(filename, df, info=None):
    try:

        with open(filename, 'a') as file:
            for i in df:
                file.write(' {}: {} |'.format(i, df[i][0]))
            for key, value in info.items():
                if key == 'Y_pred_in_test':
                    # Handling the 2D array, converting each sub-array to a string and joining them
                    array_string = '[' + '; '.join([' '.join([f"{val:.8f}" for val in row]) for row in value]) + ']'
                    file.write(f' {key}: {array_string} |')
                else:
                    file.write(f' {key}: {value} |')
            file.write('\n')

    except Exception as e:
        print("Error:", e)


def read_past_value(directory, check_metrice):
    hist_loc = directory + 'history.csv'
    if os.path.exists(hist_loc):
        history = pd.read_csv(hist_loc)
        if history.get(f'val_{check_metrice}') is not None:
            return np.max(history['val_' + check_metrice])

    return 0


def read_current_value(Y_pred, Y_true, check_metrice):
    if check_metrice == 'accuracy':
        return accuracy_score(Y_true, Y_pred)
    if check_metrice == 'sensitivity' or check_metrice == 'recall':
        return recall_score(Y_true, Y_pred)
    if check_metrice == 'f1_score':
        return f1_score(Y_true, Y_pred, average='binary')
    else:
        raise ('You have not create a calculation for: ' + check_metrice)


def check_if_save_model(output_directory, Y_pred, Y_true, check_metrice, info):

    Y_pred_binary = np.argmax(Y_pred, axis=1)
    Y_true_binary = np.argmax(Y_true, axis=1)
    
    past_metrice = read_past_value(output_directory, check_metrice)
    current_metrice = read_current_value(Y_pred_binary, Y_true_binary, check_metrice)
    hist_df_metrics = calculate_metrics(Y_true, Y_pred, 0)
    
    if current_metrice >= past_metrice:
        return True
    return False


def save_validation_acc(output_directory, Y_pred, Y_true, check_metrice, info, save_file_name='val_acc.txt'):
    Y_pred_binary = np.argmax(Y_pred, axis=1)
    Y_true_binary = np.argmax(Y_true, axis=1)
    past_metrice = read_past_value(output_directory, check_metrice)
    current_metrice = read_current_value(Y_pred_binary, Y_true_binary, check_metrice)
    hist_df_metrics = calculate_metrics(Y_true, Y_pred, 0)
    save_data_to_file(output_directory + save_file_name, hist_df_metrics, info)
    print(f'current saved file: {output_directory}' + save_file_name)
    print(f"Current {check_metrice} + {save_file_name}: {current_metrice}")

    if current_metrice > past_metrice:
        return True
    return False

def save_validation_acc_multi_task(output_directory, Y_pred, Y_true, check_metric, info, save_file_name='val_acc.txt'):
    metrics = {}
    Y_pred = np.array(Y_pred)
    for i, task_name in enumerate(check_metric.keys()):
        Y_task_i = Y_pred[i] if Y_pred.shape == 3 else Y_pred   
        Y_pred_binary = np.argmax(Y_task_i, axis=1)
        Y_true_binary = np.argmax(Y_true[:, i], axis=1)
        past_metric = read_past_value(output_directory, check_metric[task_name])
        current_metric = read_current_value(Y_pred_binary, Y_true_binary, check_metric[task_name])
        hist_df_metrics = calculate_metrics(Y_true[:, i], Y_task_i, info['duration'])
        
        save_data_to_file(output_directory + save_file_name.replace('.txt', f'_{task_name}.txt'), hist_df_metrics, info)
        print(f'Current saved file: {output_directory}' + save_file_name.replace('.txt', f'_{task_name}.txt'))
        print(f"Current {check_metric[task_name]}: {current_metric}")

        metrics[task_name] = current_metric > past_metric

    return any(metrics.values())

def save_validation_pred_regression(output_directory, info):
    save_data_to_file(output_directory + 'val_acc.txt', [], info)
    


def normalize_individual(data):
    # Iterate over each subject | optimized instead of using for
    normalized_data = np.empty_like(data)

    # if data.ndim >= 3:
    #     # For a 3D array, calculate std along the last two axes
    #     mean = np.mean(data, axis=(1,2), keepdims=True)
    #     std = np.std(data, axis=(1, 2), keepdims=True)
    # elif data.ndim == 2:
    #     # For a 2D array, calculate std along the last axis only
    #     mean = np.mean(data, axis=1, keepdims=True)
    #     std = np.std(data, axis=1, keepdims=True)
    # normalized_data = (data - mean)/std

    for i in range(data.shape[0]):
        # Calculate the mean and standard deviation for the current subject
        mean = np.mean(data[i])
        std = np.std(data[i])

        # Perform z-normalization for the current subject
        normalized_data[i] = (data[i] - mean) / std

    # mean = np.mean(data, axis=(1,2))
    # std = np.std(data, axis=(1,2))

    return normalized_data





def read_data_fnirs(file_name, model_name, hb_path, adj_path, do_individual_normalize=True, total_k=10, num_of_k=1):
    """
    normalization_method
     0 -> All sample normalization 
     1 -> Single sample normalization
    """
    using_raw_data = False
    # if model_name == 'wang_alex':
    #     # correct_channel_
    #     data = np.load(file_name + '/nor_allsubject_data.npy')
    # elif model_name == 'chao_cfnn':
    #     data = np.load(file_name + '/data.npy')  # subject, feautures
    # elif model_name == 'zhu_xgboost':
    #     data = np.load(file_name + '/raw_data.npy')  # subject, feautures
    # elif model_name == 'dgi_transformer':
    #     data = np.load(file_name + '/dgi_data.npy')
    # elif model_name == 'yu_gnn':
    #     data = np.load(file_name + '/H.npy')  # H_scaled
    # else:
    #     # correct_channel_
    #     if using_raw_data:
    #         data = np.load(file_name + '/raw_cor_ch_data.npy')
    #     else:
    #         data = np.load(file_name + '/correct_channel_data.npy')
    data = np.load(file_name + '/' + hb_path)
    # if model_name != 'dgi_transformer':
    #     data = z_norm(data, normalization_method=1)
    print('I am using nor-hbo-hc-mdd dataset, so no normalization is used. Please remember here' * 10)
    if model_name != 'chao_cfnn' and model_name != 'zhu_xgboost':
        data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))
        # if data shape is like 458, 125, 52
        # change to 458, 52, 125
        if data.shape[2] == 52:
            data = np.transpose(data, (0, 2, 1, 3))

    if do_individual_normalize:
        data = normalize_individual(data)

    label = np.load(file_name + '/label.npy')
    label = onehotEncode(label.astype(int))

    if model_name == 'comb_cnn':
        label = label.astype('float32')

    idx = np.random.permutation(data.shape[0])

    k = num_of_k
    one_fold_number = data.shape[0]//total_k
    nb_test = one_fold_number
    X_test = data[k*one_fold_number:(k+1)*one_fold_number]
    Y_test = label[k*one_fold_number:(k+1)*one_fold_number]
    X_train = np.concatenate(
        (data[0:k*one_fold_number], data[(k+1)*one_fold_number:]))
    Y_train = np.concatenate(
        (label[0:k*one_fold_number], label[(k+1)*one_fold_number:]))

    # X_test = data[idx[:nb_test]]
    # X_train = data[idx[nb_test:]]

    # Y_test = label[idx[:nb_test]]
    # Y_train = label[idx[nb_test:]]
    # print(f'X_train.shape -> {X_train.shape}')
    # print(f'Y_train.shape -> {Y_train.shape}')

    # if model_name in ['gnn_transformer', 'gnn', 'yu_gnn', 'mvg_transformer', 'mgn_transformer', 'graphsage_transformer']:
    #     if model_name == 'gnn_transformer' or model_name == 'gnn':
    #         adj = np.load(file_name + '/euclidean_matrix.npy')
    #     if model_name == 'yu_gnn':
    #         adj = np.load(file_name + '/A.npy')[...,1] # read HbO_correlation
    #     if model_name == 'mvg_transformer':
    #         adj = np.load(file_name + '/multiview_adj_matrix5.npy')
    #     if model_name == 'mgm_transformer':
    #         adj = np.load(file_name + '/multiview_adj_matrix5.npy')
    #     if model_name == 'mgn_transformer':
    #         adj = np.load(file_name + '/euclidean_mgn_matrix.npy')
    #     if model_name == 'graphsage_transformer':
    #         adj = np.load(file_name + '/multiview_adj_matrix5.npy')[...,0]
    if adj_path is not None:
        adj = np.load(file_name + '/' + adj_path)
        adj_test = adj[k*one_fold_number:(k+1)*one_fold_number]
        adj_train = np.concatenate(
            (adj[0:k*one_fold_number], adj[(k+1)*one_fold_number:]))
        return X_train, X_test, Y_train, Y_test, adj_train, adj_test
    else:
        return X_train, X_test, Y_train, Y_test


def simply_read_data_fnirs(file_name, label_path, hb_path, adj_path=None, cli_demo_path=None):

    data = np.load(file_name + '/' + hb_path)

    label = np.load(file_name + '/' + label_path)
    if label_path == 'label.npy':
        label = onehotEncode(label.astype(int))

    if adj_path is not None:
        adj = np.load(file_name + '/' + adj_path)
        if cli_demo_path is not None:
            cli_demo = np.load(file_name + '/' + cli_demo_path)
            return data, label, adj, cli_demo
        return data, label, adj
    else:
        return data, label

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

def LOOCV_CV(data, label, num, adj=None):
    X_val = data[num:num+1]
    Y_val = label[num:num+1]
    X_train = np.concatenate((data[0:num], data[num+1:]), axis=0)
    Y_train = np.concatenate((label[0:num], label[num+1:]), axis=0)

    if adj is None:
        return X_train, Y_train, X_val, Y_val
    else:
        adj_val = adj[num:num+1]
        adj_train = adj[[i for i in range(adj.shape[0]) if i != num]]   
        return X_train, Y_train, X_val, Y_val, adj_train, adj_val

"""
holdout_rate = 1/3 
so for pretreatment dataset 
    - train_val dataset is 10(1) + 34(0) * subject(label)
        - 2 + 6 
    - test dataset is 5(1) + 16(0)
for negtive label 
    - train_val dataset is 8(1) + 22(0) 
    - test dataset is 4(1) + 11(0)
"""
import time
# def shuffle_data_label(data, label, seed):
#     seed = 1720051797 # int(time.time())
#     print(f'Current seed is for shuffling: {seed}' * 10)
#     random.seed(seed)
#     combined = list(zip(data, label))
#     random.shuffle(combined)
#     data, label = zip(*combined)
#     return np.array(data), np.array(label) 

def shuffle_data_label(data, label, seed):
    # seed = 1720051797 # int(time.time())
    # print(f'Current seed is for shuffling: {seed}' * 10)
    # random.seed(seed)
    indices = np.arange(data.shape[0])
    combined = list(zip(data, label, indices))
    random.shuffle(combined)
    data, label, indices = zip(*combined)
    return np.array(data), np.array(label), np.array(indices)


def shuffle_data_demo_label(data, label, demo, seed):
    random.seed(seed)
    combined = list(zip(data, demo, label))
    random.shuffle(combined)
    data, demo, label = zip(*combined)
    return np.array(data), np.array(demo), np.array(label) 

def stratified_k_fold_cross_validation_with_holdout(data, label, k, num_of_k_fold, adj=None, seed=42, hold_out_div=3):
    total_amount = data.shape[0] 
    _, _, indices = shuffle_data_label(data, label, seed)
    
    # if label.shape is 3. 
    # It means label = (subject, type_of_label[task], onehot_encoded)   
    # if label.shape is 2 
    # It means label = (subject, onehot_encoded) with only one task
    if len(label.shape) == 3:
        label_not_onehot = np.argmax(label[:, -1, :], axis=1)
    elif len(label.shape) == 2:
        label_not_onehot = np.argmax(label, axis=1)
    else:
        mean_label = np.mean(label)
        label_not_onehot = [1 if i > mean_label else 0 for i in label]
    pos_indices = indices[label_not_onehot==1]
    neg_indices = indices[label_not_onehot==0]
    
    holdout_pos_num = pos_indices.shape[0] // hold_out_div
    holdout_neg_num = neg_indices.shape[0] // hold_out_div
    
    indices_test = np.concatenate((pos_indices[:holdout_pos_num], neg_indices[:holdout_neg_num]), axis=0)
    
    train_val_pos_indices = pos_indices[holdout_pos_num:]
    train_val_neg_indices = neg_indices[holdout_neg_num:]

    train_val_pos_num = pos_indices.shape[0]-holdout_pos_num 
    train_val_neg_num = neg_indices.shape[0]-holdout_neg_num
    one_fold_number_pos = train_val_pos_num//num_of_k_fold
    one_fold_number_neg = train_val_neg_num//num_of_k_fold
    val_pos_indices = train_val_pos_indices[k*one_fold_number_pos:(k+1)*one_fold_number_pos]
    val_neg_indices = train_val_neg_indices[k*one_fold_number_neg:(k+1)*one_fold_number_neg]
    
    indices_val = np.concatenate((val_pos_indices, val_neg_indices), axis=0)
    
    train_pos_indices = np.concatenate((train_val_pos_indices[0:k*one_fold_number_pos], train_val_pos_indices[(k+1)*one_fold_number_pos:]), axis=0)
    train_neg_indices = np.concatenate((train_val_neg_indices[0:k*one_fold_number_neg], train_val_neg_indices[(k+1)*one_fold_number_neg:]), axis=0)
    
    indices_train = np.concatenate((train_pos_indices, train_neg_indices), axis=0)
    
    X_train, X_val, X_test = data[indices_train], data[indices_val], data[indices_test]
    Y_train, Y_val, Y_test = label[indices_train], label[indices_val], label[indices_test]

    if adj is None:
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    else:
        adj_train = adj[:X_train.shape[0]]
        adj_val = adj[:X_val.shape[0]]
        adj_test = adj[:X_test.shape[0]]
        return X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test

def nested_cross_validation_split(data, label, inner_k, total_inner_k, outer_k, total_outer_k, adj=None, seed=42):
    total_amount = data.shape[0] 
    _, _, indices = shuffle_data_label(data, label, seed)
    
    # if label.shape is 3. 
    # It means label = (subject, type_of_label[task], onehot_encoded)   
    # if label.shape is 2 
    # It means label = (subject, onehot_encoded) with only one task
    if len(label.shape) == 3:
        label_not_onehot = np.argmax(label[:, -1, :], axis=1)
    elif len(label.shape) == 2:
        label_not_onehot = np.argmax(label, axis=1)
    else:
        mean_label = np.mean(label)
        label_not_onehot = [1 if i > mean_label else 0 for i in label]
    pos_indices = indices[label_not_onehot==1]
    neg_indices = indices[label_not_onehot==0]
    
    holdout_pos_num = pos_indices.shape[0] // total_outer_k
    holdout_neg_num = neg_indices.shape[0] // total_outer_k
    
    indices_test = np.concatenate((pos_indices[outer_k*holdout_pos_num:(outer_k+1)*holdout_pos_num], neg_indices[outer_k*holdout_neg_num:(outer_k+1)*holdout_neg_num]), axis=0)
    
    train_val_pos_indices = np.concatenate((pos_indices[0:outer_k*holdout_pos_num], pos_indices[(outer_k+1)*holdout_pos_num:]), axis=0)
    train_val_neg_indices = np.concatenate((neg_indices[0:outer_k*holdout_neg_num], neg_indices[(outer_k+1)*holdout_neg_num:]), axis=0)

    train_val_pos_num = train_val_pos_indices.shape[0]
    train_val_neg_num = train_val_neg_indices.shape[0]
    
    one_fold_number_pos = train_val_pos_num//total_inner_k
    one_fold_number_neg = train_val_neg_num//total_inner_k
    val_pos_indices = train_val_pos_indices[inner_k*one_fold_number_pos:(inner_k+1)*one_fold_number_pos]
    val_neg_indices = train_val_neg_indices[inner_k*one_fold_number_neg:(inner_k+1)*one_fold_number_neg]
    indices_val = np.concatenate((val_pos_indices, val_neg_indices), axis=0)
    
    train_pos_indices = np.concatenate((train_val_pos_indices[0:inner_k*one_fold_number_pos], train_val_pos_indices[(inner_k+1)*one_fold_number_pos:]), axis=0)
    train_neg_indices = np.concatenate((train_val_neg_indices[0:inner_k*one_fold_number_neg], train_val_neg_indices[(inner_k+1)*one_fold_number_neg:]), axis=0)
    indices_train = np.concatenate((train_pos_indices, train_neg_indices), axis=0)
    
    X_train, X_val, X_test = data[indices_train], data[indices_val], data[indices_test]
    Y_train, Y_val, Y_test = label[indices_train], label[indices_val], label[indices_test]

    if adj is None:
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    else:
        adj_train = adj[:X_train.shape[0]]
        adj_val = adj[:X_val.shape[0]]
        adj_test = adj[:X_test.shape[0]]
        return X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test
    
def stratified_k_fold_cross_validation_with_holdout_with_cli_demo(data, label, cli_demo, k, num_of_k_fold, adj=None, seed=42):
    data, cli_demo, label = shuffle_data_demo_label(data, label, cli_demo, seed)
    index_1 = np.where(label == 1)[0]
    index_0 = np.where(label == 0)[0]

    num_of_test_data_1 = int(len(index_1) // 3)
    num_of_test_data_0 = int(len(index_0) // 3)

    test_index_1 = index_1[:num_of_test_data_1]
    test_index_0 = index_0[:num_of_test_data_0]

    val_train_index_1 = index_1[num_of_test_data_1:]
    val_train_index_0 = index_0[num_of_test_data_0:]

    one_fold_number_1 = len(val_train_index_1) // num_of_k_fold
    one_fold_number_0 = len(val_train_index_0) // num_of_k_fold


    val_index_1 = val_train_index_1[k*one_fold_number_1:(k+1)*one_fold_number_1]
    val_index_0 = val_train_index_0[k*one_fold_number_0:(k+1)*one_fold_number_0]

    train_index_1 = np.concatenate([val_train_index_1[:k*one_fold_number_1], val_train_index_1[(k+1)*one_fold_number_1:]])
    train_index_0 = np.concatenate([val_train_index_0[:k*one_fold_number_0], val_train_index_0[(k+1)*one_fold_number_0:]])

    test_index = np.concatenate([test_index_1, test_index_0])
    val_index = np.concatenate([val_index_1, val_index_0])
    train_index = np.concatenate([train_index_1, train_index_0])

    X_train = data[train_index]
    X_val = data[val_index]
    X_test = data[test_index]

    Y_train = label[train_index]
    Y_val = label[val_index]
    Y_test = label[test_index]

    cli_demo_train = cli_demo[train_index]
    cli_demo_val = cli_demo[val_index]
    cli_demo_test = cli_demo[test_index]

    
    
    if adj is None:
        raise NotImplementedError('adj, should have adj')
        return X_train, Y_train, X_val, Y_val, X_test, Y_test, cli_demo_train, cli_demo_val, cli_demo_test
    else:
        adj_train = adj[train_index]
        adj_val = adj[val_index]
        adj_test = adj[test_index]
        return X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test, cli_demo_train, cli_demo_val, cli_demo_test
        # raise NotImplementedError('adj

def stratified_LOO_nested_CV(data, label, k, num_of_k_fold, current_loo, adj=None, cli_demo=None):
    total_amount = data.shape[0] 
    if len(np.unique(label)) > 2: 
        label_not_onehot = label.copy()
        median_label = np.median(label_not_onehot)
        bigger_median = label_not_onehot >= median_label
        smaller_median = label_not_onehot < median_label
        label_not_onehot[bigger_median] = 1
        label_not_onehot[smaller_median] = 0
    else:
        label_not_onehot = np.argmax(label, axis=1)
    print('label:', label)
    print('np.unique(label)', np.unique(label))
    print('label_not_onehot:', label_not_onehot)
    
    test_index = np.array([current_loo])
    train_val_index = [i for i in range(total_amount) if i != current_loo]
    train_val_index_1 = [i for i in train_val_index if label_not_onehot[i] == 1]
    train_val_index_0 = [i for i in train_val_index if label_not_onehot[i] == 0]
    
    train_val_pos_num = len(train_val_index_1)
    train_val_neg_num = len(train_val_index_0)
    
    one_fold_number_pos = train_val_pos_num//num_of_k_fold
    if one_fold_number_pos == 0: one_fold_number_pos = 1
    one_fold_number_neg = train_val_neg_num//num_of_k_fold
    if one_fold_number_neg == 0: one_fold_number_neg = 1
    
    val_index_1 = train_val_index_1[k*one_fold_number_pos:(k+1)*one_fold_number_pos]
    val_index_0 = train_val_index_0[k*one_fold_number_neg:(k+1)*one_fold_number_neg]
    
    train_index_1 = np.concatenate((train_val_index_1[0:k*one_fold_number_pos], train_val_index_1[(k+1)*one_fold_number_pos:]), axis=0)
    train_index_0 = np.concatenate((train_val_index_0[0:k*one_fold_number_neg], train_val_index_0[(k+1)*one_fold_number_neg:]), axis=0)
    
    val_index = np.concatenate([val_index_1, val_index_0])
    train_index = np.concatenate([train_index_1, train_index_0])
    
    train_index = tf.cast(train_index, dtype=tf.int32)
    
    X_train = data[train_index]
    X_val = data[val_index]
    X_test = data[test_index]

    Y_train = label[train_index]
    Y_val = label[val_index]
    Y_test = label[test_index]
    
    if adj is None:
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    else:
        adj_train = adj[:X_train.shape[0]]
        adj_val = adj[:X_val.shape[0]]
        adj_test = adj[:X_test.shape[0]]
        
        if cli_demo is not None:
            cli_demo_train = cli_demo[train_index]
            cli_demo_val = cli_demo[val_index]
            cli_demo_test = cli_demo[test_index]
            return X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test, cli_demo_train, cli_demo_val, cli_demo_test 
        else:
            return X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test
            
# # backup - 2024-05-09 - achieved AUC of 0.74 using fNIRS-only
# def stratified_LOO_nested_CV(data, label, k, num_of_k_fold, current_loo, adj=None):
#     total_amount = data.shape[0] 
    
#     X_test = data[current_loo:current_loo+1]
#     Y_test = label[current_loo:current_loo+1]
    
#     data = np.concatenate((data[0:current_loo], data[current_loo+1:]), axis=0)
#     label = np.concatenate((label[0:current_loo], label[current_loo+1:]), axis=0)
#     label_not_onehot = np.argmax(label, axis=1)

#     pos = data[label_not_onehot==1]
#     neg = data[label_not_onehot==0]
    
#     train_val_pos_num = pos.shape[0]
#     train_val_neg_num = neg.shape[0]
#     one_fold_number_pos = train_val_pos_num//num_of_k_fold
#     if one_fold_number_pos == 0: one_fold_number_pos = 1
#     one_fold_number_neg = train_val_neg_num//num_of_k_fold
#     if one_fold_number_neg == 0: one_fold_number_neg = 1

#     train_val_pos = pos
#     train_val_neg = neg

#     val_pos = train_val_pos[k*one_fold_number_pos:(k+1)*one_fold_number_pos]
#     val_neg = train_val_neg[k*one_fold_number_neg:(k+1)*one_fold_number_neg]
    
#     X_val = np.concatenate((val_pos, val_neg), axis=0)
#     Y_val = np.concatenate((np.ones(val_pos.shape[0]), np.zeros(val_neg.shape[0])), axis=0)
    
#     train_pos = np.concatenate((train_val_pos[0:k*one_fold_number_pos], train_val_pos[(k+1)*one_fold_number_pos:]), axis=0)
#     train_neg = np.concatenate((train_val_neg[0:k*one_fold_number_neg], train_val_neg[(k+1)*one_fold_number_neg:]), axis=0)
    
#     X_train = np.concatenate((train_pos, train_neg), axis=0)
#     Y_train = np.concatenate((np.ones(train_pos.shape[0]), np.zeros(train_neg.shape[0])), axis=0)
    
#     Y_train, Y_val = onehotEncode(Y_train).astype('float32'), onehotEncode(Y_val).astype('float32')
#     if adj is None:
#         return X_train, Y_train, X_val, Y_val, X_test, Y_test
#     else:
#         adj_train = adj[:X_train.shape[0]]
#         adj_val = adj[:X_val.shape[0]]
#         adj_test = adj[:X_test.shape[0]]
#         return X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test

def calculate_metrics(y_true, y_pred, duration, y_true_onehot=None, y_pred_onehot=None):

    Y_pred_binary = np.argmax(y_pred, axis=1)
    Y_true_binary = np.argmax(y_true, axis=1)
    

    save_metrices = ['AUC', 'accuracy', 'sensitivity',
                        'specificity', 'F1-score', 'duration']
    # res = pd.DataFrame(data=np.zeros((1, len(save_metrices)), dtype=np.float), index=[0],
    #                    columns=save_metrices)
    res = pd.DataFrame(data=np.zeros((1, len(save_metrices)),
                       dtype=np.float64), index=[0], columns=save_metrices)
    res['accuracy'] = round(accuracy_score(Y_true_binary, Y_pred_binary), 5)

    res['sensitivity'] = round(recall_score(Y_true_binary, Y_pred_binary), 5)

    res['specificity'] = round(get_specificity(Y_true_binary, Y_pred_binary), 5)

    # For F1 Y_true_binary
    f1 = f1_score(Y_true_binary, Y_pred_binary, average='weighted')
    res['F1-score'] = round(f1, 5)

    auc = roc_auc_score(Y_true_binary, y_pred[:, 1])
    res['AUC'] = round(auc, 5)
    
    res['duration'] = round(duration, 5)
    return res


# 数据的格式需要保持不变
# output_directory[0:-17] 作为保存图片的位置
def save_logs(model, output_directory, result_name, hist, y_pred, y_true, duration, lr=True, is_saving_checkpoint=False, hyperparameters=None, y_true_onehot=None, y_pred_onehot=None, pass_history=False):
    # save best model of all:

    if hyperparameters is not None:
        with open(output_directory + "best_hyperparameters.txt", "w") as file:
            file.write("hyperparameters = {\n")
            for key, value in hyperparameters.items():
                file.write(f"    {key} = {value} \n")
            file.write("}")
    print(f"save_checkpoiont - output_directory: {is_saving_checkpoint} {output_directory}")

    # saving checkpoint should not be working because we want the result to be more objective (!!!in test or validation should be careful!!!)
    if is_saving_checkpoint:
        model.save_weights(output_directory + 'fold-best-checkpoint')
    else:
        pass
        # save_model(model, output_directory + 'fold-best-model.keras')
    if pass_history != True:
        hist_df = pd.DataFrame(hist.history)
        hist_df.to_csv(output_directory + 'history.csv', index=False)

    hist_df_metrics = calculate_metrics(
        y_true, y_pred, duration, y_true_onehot, y_pred_onehot)
    hist_df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

def save_logs_regression(model, output_directory, hist, is_saving_checkpoint=False):
    # save best model of all:

    if is_saving_checkpoint:
        model.save_weights(output_directory + 'fold-best-checkpoint')
    else:
        pass
        # save_model(model, output_directory + 'fold-best-model.keras')
    
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # it is really import to cast into tf.float 32 to train the models
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}

def sinusoidal_lr(epoch, lr):
    base_lr = 1e-5
    amplitude = 1e-5
    new_lr = base_lr + amplitude * math.sin(2 * math.pi * epoch / 50)
    return new_lr

def train_model_using_loocv(data, label, model):
    loo = LeaveOneOut()
    result = []

    # Loop over each train/test split
    for train_index, test_index in loo.split(data):
        # Split the data into training and testing sets
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]
        
        # Train the classifier
        model.fit(X_train, y_train)

        # Predict the label for the test set
        y_pred = model.predict(X_test)

        # Append the accuracy to the list
        result.append([y_pred, y_test])

    return np.array(result), model

def print_md_table(model_name, set, metrics):
    print()
    print('| Model Name | Val/Test Set | Accuracy | Sensitivity | Specificity | F1 Score |')
    print('|------------|--------------|----------|-------------|-------------|----------|')
    print(f'| {model_name} | {set} |', end = '')
    for i in range(4):
        print(f" {metrics[i]:.4f} |", end = '')
    print()
    print(''*10)
    

def get_metrics(y_true, y_pred):
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 明确指定labels参数
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 现在cm是一个2x2矩阵，即使数据只包含一个类别
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)

    return accuracy, sensitivity, specificity, f1


def save_current_file_to_folder(save_file_paths, destination_folder):
    for current_file_path in save_file_paths:
        try:
            os.makedirs(destination_folder, exist_ok=True)
            
            # Define the destination path
            destination_path = os.path.join(destination_folder, 'used_' + os.path.basename(current_file_path))
            
            # Debugging: Print the destination path
            # print(f"Destination path: {destination_path}")
            
            # Copy the current file to the destination folder
            shutil.copy(current_file_path, destination_path)
            
            print(f"File copied to: {destination_path}")
        except Exception as e:
            print(f"An error occurred: {e} when save_current_file_to_folder")
        
        
def plot_evaluation_metrics_header( table_name = 'Model Name', parameter_name=None, val_auc_threshold=0):
    row_1 = f'| {table_name} | Testing Set |             |             |             | Validation Set |             |             |      Threshold={val_auc_threshold}       |             |'
    row_2 = '|------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|'
    row_3 = f'|      {parameter_name}      | bAcc(%) | Sen(%) | Spe(%) | AUC(%) | bAcc(%) | Sen(%) | Spe(%) | AUC(%) | Duration(s) |'
    
    for i in [row_1,row_2, row_3]:
        print(i)
        
        
        
def save_hist_file(hist, output_directory):
    
    # Assume hist is an object that has a history attribute which is a dictionary of the training history
    hist_df = pd.DataFrame(hist.history)

    history_file = os.path.join(output_directory, 'history.csv')

    # Check if the history.csv file exists
    if os.path.exists(history_file):
        # Read the existing CSV file into a DataFrame
        existing_df = pd.read_csv(history_file)
        
        # Concatenate the existing DataFrame with the new hist_df
        combined_df = pd.concat([existing_df, hist_df], ignore_index=True)
    else:
        # If the file does not exist, the combined DataFrame is the new hist_df
        combined_df = hist_df

    # Save the combined DataFrame back to the history.csv file
    combined_df.to_csv(history_file, index=False)
    
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)    