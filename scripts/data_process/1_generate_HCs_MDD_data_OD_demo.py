# packages 
# load 

import sys
import glob
import os 
import time
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
import pingouin as pg
import subprocess

import os
# save the data
output_fold = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/diagnosis'
if not os.path.exists(output_fold):
    os.makedirs(output_fold)

"""

input: .csv file of individual

Output: (1251, 52) data shape 

"""
def read_hb_from_file(example_path): # Open the file and read through the first few lines to find where the data starts
    with open(example_path, 'r') as file:
        lines = file.readlines()
        data_start_line = None
        for i, line in enumerate(lines):
            if 'Data' in line:  # This should match the unique identifier of the data section
                data_start_line = i + 1
                # print(lines[data_start_line])
                break

    if data_start_line is not None:

        # Read the data section, skipping the lines of the metadata
        data = pd.read_csv(example_path, skiprows=data_start_line)

        # Now you have metadata and data as separate DataFrames
        # print(data)
    else:
        print("Data section not found.")
        
    np_data = data.to_numpy()
    ch_data = np_data[:, 1:1+52]

    return ch_data


""" Loop the demographic file and find all the subjects with available hb data

input: .csv file of demographic information

output: list of subjects with available hb data
"""
def loop_find_subs_with_HB(demographic_filepath, hb_filepath):
    excel_data = pd.read_excel(demographic_filepath, sheet_name='Summary T0T8_fNIRS Analysis')
    HCS_path = hb_filepath + '/Controls/MES'
    MDD_path = hb_filepath + '/Patients/MES'

    TOTAL_AMOUNT_HCS = 0 
    TOTAL_AMOUNT_MDD = 0

    HCS_LIST = []
    MDD_LIST = []
    # Make sure data exist both in excel and having light intensity data
    for subject_name in excel_data['Subject ID'][2:]:
        if subject_name[:2] == 'CT':
            deoxy = os.path.join(HCS_path, subject_name + '*' + '.csv')
        elif subject_name[:2] == 'PT':
            deoxy = os.path.join(MDD_path, subject_name + '*' + '.csv')
        file_list = glob.glob(deoxy)
        print('deoxy ->', file_list)

        if len(file_list) >= 1:
            if subject_name[:2] == 'CT':
                TOTAL_AMOUNT_HCS += 1
                HCS_LIST.append(subject_name)
            if subject_name[:2] == 'PT':
                TOTAL_AMOUNT_MDD += 1
                MDD_LIST.append(subject_name)
    print('Total amount of HCS:', TOTAL_AMOUNT_HCS)
    print('Total amount of MDD:', TOTAL_AMOUNT_MDD)
    print('HCs_list:', HCS_LIST)
    print('MDD_list:', MDD_LIST)

    return HCS_LIST, MDD_LIST


def get_file_name(path, rest):
    file_pattern = os.path.join(path, rest)
    file_list = glob.glob(file_pattern)
    return file_list


def read_demographic_data(demographic_filepath, loop_list):
    excel_data = pd.read_excel(demographic_filepath, sheet_name='Summary T0T8_fNIRS Analysis')
    return_demographic_data = []
    for sub_id in loop_list:
        sub = excel_data['Subject ID'] == sub_id
        demographic = excel_data[sub]
        demographic_selected = demographic.iloc[:, 2:13]
        return_demographic_data.append(demographic_selected.values[0])
    return_demographic_data = np.array(return_demographic_data)
    return return_demographic_data



def compute_delta(data):
    data = np.nan_to_num(data, 0)
    light_700 = data[:,0::2].tolist()
    light_830 = data[:,1::2].tolist()
    return np.array([light_700, light_830])

def read_od_from_file(example_path): # Open the file and read through the first few lines to find where the data starts
    with open(example_path, 'r') as file:
        lines = file.readlines()
        data_start_line = None
        for i, line in enumerate(lines):
            if 'Data' in line:  # This should match the unique identifier of the data section
                data_start_line = i + 1
                # print(lines[data_start_line])
                break

    if data_start_line is not None:

        # Read the data section, skipping the lines of the metadata
        data = pd.read_csv(example_path, skiprows=data_start_line)

        # Now you have metadata and data as separate DataFrames
        # print(data)
    else:
        print("Data section not found.")
        
    np_data = data.to_numpy()
    ch_data = np_data[:, 1:1+52*2]
    
    od_data = compute_delta(ch_data)

    return od_data


def read_od_data(file_path, loop_list):
    all_subject_od = None #np.empty((len(loop_list), 2, 52, TOTAL_timepoint)) # 2 is the hbo and hbr intensities
    for index, sub_id in enumerate(loop_list):
        od_path = glob.glob(os.path.join(file_path, sub_id + '*' + '.csv'))
        if len(od_path)<1 :
            print(sub_id)
            print(od_path)
            print('-----------------------------------')
            
        od = read_od_from_file(od_path[0])
        if all_subject_od is None:
            all_subject_od = np.empty((len(loop_list), 2, od.shape[-2], 52))
        print('od:', od.shape)
        all_subject_od[index, :, :, :] = od

    return all_subject_od



def read_hb_data(file_path, loop_list):
    all_subject_hb = np.empty((len(loop_list), 52, TOTAL_timepoint, 3))
    for index, sub_id in enumerate(loop_list):
        hbo_path = glob.glob(os.path.join(file_path, sub_id + '*' + 'Oxy.csv'))
        hbr_path = glob.glob(os.path.join(file_path, sub_id + '*' + 'Deoxy.csv'))
        if len(hbo_path)<1 or len(hbr_path)<1:
            print(sub_id)
            print(hbo_path)
            print(hbr_path)
            print('-----------------------------------')
            
        hbo = read_hb_from_file(hbo_path[0])
        hbr = read_hb_from_file(hbr_path[0])
        all_subject_hb[index, :, :, 0] = hbo.T
        all_subject_hb[index, :, :, 1] = hbr.T
        all_subject_hb[index, :, :, 2] = hbo.T + hbr.T
    return all_subject_hb
def remove_nan_for_demographic_data(demographic_data):
    for index, sub_value in enumerate(demographic_data):
        try: 
            demographic_data[index] = sub_value.astype(int)
            pass
        except:
            for i, v in enumerate(sub_value):
                if type(v) is not int:
                    print('index:', i, 'value:', v)
                    sub_value[i] = 1
                    print('index:', i, 'fixed - value:', sub_value[i])
                    print('there should be a subject whose handedness is empty, the above steps set it to be 1')
            demographic_data[index] = sub_value.astype(int)

    demographic_data = demographic_data.astype(int)
    return demographic_data

def get_data(demographic_filepath, hb_filepath):

    HCS_LIST, MDD_LIST = loop_find_subs_with_HB(demographic_filepath, hb_filepath)

    # read the demographic data
    HCS_demographic = read_demographic_data(demographic_filepath, HCS_LIST)
    MDD_demographic = read_demographic_data(demographic_filepath, MDD_LIST)
    demographic_data = np.concatenate((HCS_demographic, MDD_demographic), axis=0)
    demographic_data = remove_nan_for_demographic_data(demographic_data)
    # read the hb data shape (140, 52, 1251, 3)
    HCS_hb_data = read_od_data(hb_filepath + '/Controls/MES', HCS_LIST)
    MDD_hb_data = read_od_data(hb_filepath + '/Patients/MES', MDD_LIST)
    hb_data = np.concatenate((HCS_hb_data, MDD_hb_data), axis=0)

    label_data = np.concatenate((np.zeros(len(HCS_LIST)), np.ones(len(MDD_LIST))), axis=0)
    print(label_data.shape)
    # final return  
    # demographic data -> 140, 11 
    # Age (years)	
    # Sex	
    # Ethnicity	
    # Handedness	
    # Education (years)	
    # Current/highest education level —  Coded	
    # Perceived social support	
    # Past trauma	
    # Current psychiatric comorbidities — Binary	
    # Current psychiatric comorbidities — Coded	
    # Family history of psychiatric illness
    print('demographic data ->', demographic_data.shape)
    # hemoglobin data -> 140, 52, 1251, 3 (HbO, HbR, HbT)
    print('hemoglobin data ->', hb_data.shape)
    # label data -> 140, 
    print('label data ->', label_data.shape)

    return demographic_data, hb_data, label_data

def main():
    demographic_filepath = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/RawData/fNIRS x MDD Data_Demographics_Clinical.xlsx'
    hb_filepath = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/RawData//Baseline_fnirs'
    
    
    # get the demograophic data, hb data and label data
    demographic_data, od_data, label_data = get_data(demographic_filepath, hb_filepath)
    
    print('demographic data:', demographic_data.shape)
    print('Ligh intensity data ->', od_data.shape)
    # reshape the 140, 2, 1701, 52 into 140, 52, 2, 1701
    od_data = np.transpose(od_data, (0,3, 1, 2))
    print('od_data -> ', od_data.shape)
    
    print('label data:', label_data.shape)
    
    # save data
    import scipy.io as sio
    sio.savemat(os.path.join(output_fold, 'od_data.mat'), {'od_data': od_data})
    sio.savemat(os.path.join(output_fold, 'age_data.mat'), {'age_data': demographic_data[:,0]})


if __name__ == '__main__':
    main()
    