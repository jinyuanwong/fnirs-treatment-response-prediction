# Remember to install chord-6.0.1
from chord import Chord
# load 
import sys
import time
import os 
from sklearn.model_selection import cross_val_score,train_test_split
from datetime import date
import numpy as np
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import subprocess




def normalize_individual(data):
    # Iterate over each subject | optimized instead of using for
    normalized_data = np.empty_like(data)
    for i in range(data.shape[0]):
        # Calculate the mean and standard deviation for the current subject
        mean = np.mean(data[i])
        std = np.std(data[i])

        # Perform z-normalization for the current subject
        normalized_data[i] = (data[i] - mean) / std
        
    # mean = np.mean(data, axis=(1,2))
    # std = np.std(data, axis=(1,2))
    
    return normalized_data

def read_hb(datapath, labelpath):
    
    data = np.load(datapath)
    data = np.transpose(data, (0,2,1,3,4))

    label = np.load(labelpath)
    change = label[:,1] - label[:,0]

    hbo = data[:,:,:,0,:]
    hbr = data[:,:,:,1,:]

    hbo_base = normalize_individual(hbo[...,0])
    hbo_t8 = normalize_individual(hbo[...,1])


    hbr_base = normalize_individual(hbr[...,0])
    hbr_t8 = normalize_individual(hbr[...,1])
    return hbo_base, hbo_t8, hbr_base, hbr_t8


def check_and_create_folder(path):
    """
    Check if a folder exists at the specified path. If not, create the folder.
    
    Parameters:
    - path: The path of the folder to check and possibly create.
    
    Returns:
    - None
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder created at: {path}")
    else:
        print(f"Folder already exists at: {path}")

def balance_label_0and1(data, label):
    # balance the distribution of recover and unrecover data
    recover_data = data[label==1]
    unrecover_data = data[label==0]
    recover_shape = data[label==1].shape
    unrecover_shape = data[label==0].shape
    ration = unrecover_shape[0] / recover_shape[0]
    i_recover = 0
    i_unrecover = 0
    new_data = []
    new_label = []
    '''
    Assumption:
        1. The number of recover data is less than unrecover data

    Main logic:
        0. loop the unrecover data
        1. If the index of (recover+1)*ration < i_unrecover, then add one recover data
        2. Add the unadded recover data
    '''
    for i_unrecover in range(unrecover_shape[0]):
        new_data.append(unrecover_data[i_unrecover])
        new_label.append(0)
        if (i_recover+1)*ration < i_unrecover:
            new_data.append(recover_data[i_recover])
            new_label.append(1)
            i_recover += 1

    # 2. Add the unadded recover data
    for left_recover in range(i_recover, recover_shape[0]):
        new_data.append(recover_data[left_recover])
        new_label.append(1)
        
    new_data = np.array(new_data)
    new_label = np.array(new_label)
    return new_data, new_label

def generate_response_label(scores):
    base_time_score = scores[:,0]
    T8_time_score = scores[:,1]

    score_change_percentage = (scores[:,1]-scores[:,0])/scores[:,0] # score changes

    treatment_response_label = np.empty_like(score_change_percentage)

    for i in range(len(score_change_percentage)):
        if score_change_percentage[i] <= -0.5:
            treatment_response_label[i] = 1
        else:
            treatment_response_label[i] = 0
    return treatment_response_label

def read_basetime_hb(datapath, labelpath):
    data = np.load(datapath)
    data = np.transpose(data, (0,2,1,3))

    hbo = data[:,:,:,0]
    hbr = data[:,:,:,1]

    hbo_base = normalize_individual(hbo)
    hbr_base = normalize_individual(hbr)
    return hbo_base, hbr_base

mainfold = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/prognosis'
# data and label path
pre_and_post_datapath = mainfold + '/base_t8_data.npy'
pre_and_post_labelpath= mainfold + '/base_t8_label.npy'

# output path for pre-treatment data
output_pre_treatment_path = mainfold + '/pre_treatment_hamd_reduction_50/'

# output path for post-treatment data
output_pre_and_post_treatment_path = mainfold + '/pre_post_treatment_hamd_reduction_50/'



# read the hbo and hbr in pre-treatment time and post treatment time 
hbo_base, hbo_t8, hbr_base, hbr_t8 = read_hb(pre_and_post_datapath, pre_and_post_labelpath)

# Concatenate HbO and HbR of pre-treatment data
hbo_hbr_base = np.concatenate((hbo_base, hbr_base), axis=2)
hbo_hbr_t8 = np.concatenate((hbo_t8, hbr_t8), axis=2)
hbo_hbr_base_t8 = np.concatenate((hbo_hbr_base[...,np.newaxis], hbo_hbr_t8[...,np.newaxis]), axis=3)

"""
Here define the label of treatment response: HAMD reduction >= 50% as 1, else 0
"""
# Read the label 
scores = np.load(pre_and_post_labelpath)
treatment_response_label = generate_response_label(scores)

print(treatment_response_label)

# balance the distribution of recover and unrecover data    
pre_data, label = balance_label_0and1(hbo_hbr_base, treatment_response_label)
pre_post_data, label = balance_label_0and1(hbo_hbr_base_t8, treatment_response_label)

check_and_create_folder(output_pre_and_post_treatment_path)
np.save(output_pre_and_post_treatment_path + 'data.npy', pre_post_data)
np.save(output_pre_and_post_treatment_path + 'label.npy', label)





pre_and_post_datapath = mainfold + '/base_t8_data.npy'
pre_and_post_labelpath= mainfold + '/base_t8_label.npy'

# output path for pre-treatment data
output_pre_treatment_path = mainfold + '/pre_treatment_hamd_reduction_50/'

# output path for post-treatment data
output_pre_and_post_treatment_path = mainfold + '/pre_post_treatment_hamd_reduction_50/'


pre_datapath = mainfold + '/base_data.npy'
pre_labelpath= mainfold + '/base_label.npy'
# read the hbo and hbr in pre-treatment time and post treatment time 
hbo_base, hbr_base = read_basetime_hb(pre_datapath, pre_labelpath)

# Concatenate HbO and HbR of pre-treatment data
hbo_hbr_base = np.concatenate((hbo_base, hbr_base), axis=2)

"""
Here define the label of treatment response: HAMD reduction >= 50% as 1, else 0
"""
# Read the label 
scores = np.load(pre_labelpath)
treatment_response_label = generate_response_label(scores)

# balance the distribution of recover and unrecover data    
pre_data, label = balance_label_0and1(hbo_hbr_base, treatment_response_label)

# save file 
check_and_create_folder(output_pre_treatment_path)
np.save(output_pre_treatment_path + 'data.npy', pre_data)
np.save(output_pre_treatment_path + 'label.npy', label)