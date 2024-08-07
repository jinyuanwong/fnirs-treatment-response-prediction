# load the pretreatment data 
from scipy.stats import zscore
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import os
print(f'Current directory: {os.getcwd()}')

from utils.utils_mine import train_model_using_loocv
from utils.utils_mine import get_metrics
from utils.utils_mine import print_md_table
import time 
def start():

    pretreatment_path = 'allData/prognosis_mix_hb/pretreatment_response/merge_psychiatry_demo_dose_data.npy' #'allData/prognosis/pre_treatment_hamd_reduction_50/hb_data.npy'
    label_path = 'allData/prognosis_mix_hb/pretreatment_response/label.npy'
    # demo_path = 'allData/prognosis/pre_treatment_hamd_reduction_50/demographic_data.npy'
    pretreatment_data = np.load(pretreatment_path)#[:, -10:]
    pretreatment_label = np.load(label_path)
    # pretreatment_demo = np.load(demo_path, allow_pickle=True)
    print(f'pretreatment_data shape: {pretreatment_data.shape}')
    print(f'pretreatment_label shape: {pretreatment_label.shape}') 


    print(f'Take average of every 10 timepoints to see the result or same performance can be reproduced')
    print(pretreatment_data.shape)
    # avg_pretreatment_data = pretreatment_data[...,:-2]
    # avg_pretreatment_data = np.reshape(avg_pretreatment_data, (avg_pretreatment_data.shape[0], 52, -1, 10))
    # avg_pretreatment_data = np.mean(avg_pretreatment_data, axis=-1)
    # avg_pretreatment_data = np.reshape(avg_pretreatment_data, (avg_pretreatment_data.shape[0], -1))
    avg_pretreatment_data = pretreatment_data
    
    for i in range(1000):
        seed = int(time.time()) + i #1710222578 # it is able to replicate the model 

        
        print(f'Define model Decision Tree with default setting and seed {seed}')
        print(f'current seed: {seed}')
        model = DecisionTreeClassifier()
        model.random_state = seed

        result,model = train_model_using_loocv(avg_pretreatment_data, pretreatment_label, model)
        res_metrics = get_metrics(result[:, 1], result[:, 0])
        print_md_table('Decision Tree', 'test', res_metrics)
