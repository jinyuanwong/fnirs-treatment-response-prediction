# load the pretreatment data 
from scipy.stats import zscore
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import time
import os
print(f'Current directory: {os.getcwd()}')

from utils.utils_mine import train_model_using_loocv
from utils.utils_mine import get_metrics
from utils.utils_mine import print_md_table
def start():
    pretreatment_path = 'allData/diagnosis/hb_data.npy'
    label_path = 'allData/diagnosis/label.npy'
    demo_path = 'allData/diagnosis/demographic_data.npy'
    pretreatment_data = np.load(pretreatment_path)
    pretreatment_label = np.load(label_path)
    pretreatment_demo = np.load(demo_path, allow_pickle=True)
    print(f'pretreatment_data shape: {pretreatment_data.shape}')
    print(f'pretreatment_label shape: {pretreatment_label.shape}') 


    print(f'Take average of every 10 timepoints to see the result or same performance can be reproduced')
    print(pretreatment_data.shape)
    avg_pretreatment_data = pretreatment_data[...,:-1,0]
    print(f'Take only the HbO data -> shape: {avg_pretreatment_data.shape}')
    # avg_pretreatment_data = np.reshape(avg_pretreatment_data, (avg_pretreatment_data.shape[0], 52, -1))
    avg_pretreatment_data = np.reshape(avg_pretreatment_data, (avg_pretreatment_data.shape[0], 52, -1, 10))
    avg_pretreatment_data = np.mean(avg_pretreatment_data, axis=-1)
    print(f'Average every 10 timepoints -> shape: {avg_pretreatment_data.shape}')
    avg_pretreatment_data = np.reshape(avg_pretreatment_data, (avg_pretreatment_data.shape[0], -1))
    print(f'make it 2D -> shape: {avg_pretreatment_data.shape}')
    seed = int(time.time())
    print(f'Define model Decision Tree with default setting and seed {seed}')
    print(f'current seed: {seed}')
    model = DecisionTreeClassifier()
    model.random_state = seed

    result,model = train_model_using_loocv(avg_pretreatment_data, pretreatment_label, model)
    res_metrics = get_metrics(result[:, 1], result[:, 0])
    print_md_table('Decision Tree', 'test', res_metrics)
