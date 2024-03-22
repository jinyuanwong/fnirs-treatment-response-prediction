"""

Extract feature from time series data of Hemoglobin to classify HCs and MDD.


"""


# load the pretreatment data 
from scipy.stats import zscore
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import time
import os
from utils.utils_mine import train_model_using_loocv
from utils.utils_mine import get_metrics
from utils.utils_mine import print_md_table
from scipy.signal import welch
import pywt
from scipy.stats import kurtosis
from scipy.stats import skew
from xgboost import XGBClassifier
import pandas as pd 


from utils.fnirs_utils import AutoML_data_dict_for_DEMO
from utils.fnirs_utils import generate_random_params
from utils.fnirs_utils import specify_model_and_train
from utils.fnirs_utils import save_autodl
import random




def get_avg_pretreatment_data_label_demo_for_demo(amount_demographic):
    FOLD = 'allData/diagnosis/'
    label_path = FOLD + 'label.npy'
    demo_path = FOLD + 'demographic_data.npy'
    pretreatment_label = np.load(label_path)
    pretreatment_demo = np.load(demo_path, allow_pickle=True)

    pretreatment_demo = zscore(pretreatment_demo, axis=0)
    pretreatment_demo = pretreatment_demo[:, :amount_demographic]
    
    return pretreatment_demo, pretreatment_label

def start():
    pretreatment_demo, pretreatment_label  = get_avg_pretreatment_data_label_demo_for_demo(amount_demographic=11)

    seed = 1710772829
    
    specify_model_and_train(pretreatment_demo, pretreatment_label, 'Decision Tree', seed)

def automl(model_name, amount_demographic, save_fold, iteration_time):
    best_res_metrics, best_seed, best_para  = [0,0,0,0], 0, ''

    pretreatment_demo, pretreatment_label = get_avg_pretreatment_data_label_demo_for_demo(amount_demographic)
    
    for itr in range(iteration_time):
        seed = int(time.time())
        res_metrics, para, _ = specify_model_and_train(pretreatment_demo, pretreatment_label, model_name, seed)
        if res_metrics[-1] >= best_res_metrics[-1]:
            best_res_metrics = res_metrics
            best_seed = seed
            best_para = para
    
    
    DATA_to_save = AutoML_data_dict_for_DEMO(model_name, best_res_metrics, best_seed, iteration_time, best_para, amount_demographic)
    save_autodl(DATA_to_save, save_fold + '/DEMO_diagnosis.csv')

def get_best_seed_from_automl(file_path, ref_para):
    data = pd.read_csv(file_path)
    max_sen = np.max(data[ref_para])
    loc_data = data[data[ref_para] == max_sen]
    print('Succefully get the best model, hb_type, seed from AutoML')
    return loc_data['model'].values[0], loc_data['amount_demographic'].values[0], loc_data['seed'].values[0]

def predict_based_on_automl(csv_file='results/ML_results/AutoML/DEMO_diagnosis.csv', ref_param='F1_score'):

    # print(f'Average every 10 timepoints -> shape: {avg_pretreatment_data.shape}')
    model_name, amount_demographic, seed = get_best_seed_from_automl(csv_file, ref_param)

    pretreatment_demo, pretreatment_label = get_avg_pretreatment_data_label_demo_for_demo(amount_demographic)

    res_metrics, para, model = specify_model_and_train(pretreatment_demo, pretreatment_label, model_name, seed)
    
    return res_metrics, para, model



# def best_result():
#     pretreatment_path = 'allData/prognosis/hb_data.npy'
#     label_path = 'allData/prognosis/label.npy'
#     demo_path = 'allData/prognosis/demographic_data.npy'
#     pretreatment_data = np.load(pretreatment_path)
#     pretreatment_label = np.load(label_path)
#     pretreatment_demo = np.load(demo_path, allow_pickle=True)
#     print(f'pretreatment_data shape: {pretreatment_data.shape}')
#     print(f'pretreatment_label shape: {pretreatment_label.shape}') 

#     avg_pretreatment_data = pretreatment_data[...,:-1,0]
#     avg_pretreatment_data = avg_every_ten_point_in_last_dimension(avg_pretreatment_data)
#     print(f'Take average of every 10 timepoints to see the result or same performance can be reproduced | Take only the HbO data -> shape: {avg_pretreatment_data.shape}')
#     avg_pretreatment_data = wang_alex_feature_selection(avg_pretreatment_data, index_task_start=10, index_task_end=70, fs=1)
#     avg_pretreatment_data = np.reshape(avg_pretreatment_data, (avg_pretreatment_data.shape[0], -1))
#     seed = 1710743850
#     specify_model_and_train(avg_pretreatment_data, pretreatment_label, 'Decision Tree', seed)

"""
Some findings 

"""