# load 

import sys
import glob
sys.path.append('/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction')

import time
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import cross_val_score,train_test_split
from utils.utils_mine import*
import tensorflow as tf
import tensorflow.keras as keras
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
import collections

# path of data 


def read_from_file(example_path): # Open the file and read through the first few lines to find where the data starts
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

def get_file_name(path, rest):
    file_pattern = os.path.join(path, rest)
    file_list = glob.glob(file_pattern)
    return file_list

def check_replicate_subject(all_subject):
    all_subject_index = [i[3:] for i in all_subject]
    all_subject_index = np.array(all_subject_index).astype(int)
    print(f'all_subject_index -> {all_subject_index}')
    print('all_subject_index.shape -> ', all_subject_index.shape)
    print('len(np.unique(all_subject_index)) -> ', len(np.unique(all_subject_index)))
    is_replicated = len(np.unique(all_subject_index)) != len(all_subject_index)
    print(f"Is there any replicated number in all_subject_index? {is_replicated}")
    if is_replicated:
        replicated_elements = [item for item, count in collections.Counter(all_subject_index).items() if count > 1]
        replicated_indices = np.where(np.isin(all_subject_index, replicated_elements))[0]
        print(f" Element {replicated_elements} shows up in the following indices: {replicated_indices}")
    print(f' now will return replicated_indices[0::2]')
    return replicated_indices[0::2]


follow_up_fold = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/RawData'
T8_path = follow_up_fold + '/T8_fnirs/Session 2_VFT'
base_patient_path = follow_up_fold + '/Baseline_fnirs/Patients'
cli_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/fNIRS x MDD Data_Demographics_Clinical.xlsx'

cgi_sgs_data = pd.read_excel(cli_path, sheet_name='SDS_CGI_All Timepoints')

# cgi_sgs_data.iloc[:, 1:7]

excel_data = pd.read_excel(cli_path, sheet_name='Summary T0T8_fNIRS Analysis')
# cgi_sgs_data = pd.read_excel(cgi_sgs_path, sheet_name='SDS_CGI_All Timepoints')
label_hamd = []
demografic_data = []
baseline_clinical_data = []


for hb in ['_Oxy.csv', '_Deoxy.csv']:
    tmp = 0
    all_subject = []
    for i in os.listdir(base_patient_path):
        if i[-len(hb):] == hb:
            subject = i.split(' ')[0]
            all_subject.append(subject)
            file_pattern = os.path.join(base_patient_path, subject+'*'+hb)
            file_list = glob.glob(file_pattern)
            if len(file_list) < 1:
                print(file_list)
            tmp+=1
    all_subject.sort()
    print(f'all_subject -> {all_subject}')
print(f'all_subject -> {len(all_subject)}')

# def get_file_name(path, rest):
#     file_pattern = os.path.join(path, rest)
#     file_list = glob.glob(file_pattern)
#     return file_list
# # according to the subject name of all_subject create array now 

mdd_subject_base = []#np.zeros((len(all_subject), 1251, 52, 2)) # time, channel, hbo/hbr
all_involve_subject = []
for sub_index, subject in enumerate(all_subject):
    hamd_of_id_t1 = excel_data[excel_data['Subject ID'] == subject]['HAM-D Questionnaire (T1)'].iloc[0]
    hamd_of_id_t8 = excel_data[excel_data['Subject ID'] == subject]['HAM-D Questionnaire (T8)'].iloc[0]
    demographic = excel_data[excel_data['Subject ID'] == subject].iloc[:, 2:13]
    clinical = cgi_sgs_data[cgi_sgs_data['Subject ID'] == subject].iloc[:, 1:7]
    if type(hamd_of_id_t8) is not int:
        print(hamd_of_id_t8)
        continue
    all_involve_subject.append(subject)
    sub_label = [hamd_of_id_t1, hamd_of_id_t8]
    label_hamd.append(sub_label)
    demografic_data.append(demographic)
    baseline_clinical_data.append(clinical)
    hbo_hbr = np.zeros((1251, 52, 2))
    for hb_index, hb in enumerate(['_Oxy.csv', '_Deoxy.csv']):

        base_hb_file = get_file_name(base_patient_path, subject+'*'+hb)
        base_hb = read_from_file(base_hb_file[0])
        hbo_hbr[...,hb_index] = base_hb
    mdd_subject_base.append(hbo_hbr)
mdd_subject_base = np.array(mdd_subject_base)
label_hamd = np.array(label_hamd)
demografic_data = np.squeeze(np.array(demografic_data))
baseline_clinical_data = np.squeeze(np.array(baseline_clinical_data))


# check if there is any replicated subject, becasue there might be two files with same subject names
replicated_indices = check_replicate_subject(all_subject)
print(f'return replicated_indices {replicated_indices}')


# delete the replicated subject
mdd_subject_base = np.delete(mdd_subject_base, replicated_indices, axis=0)
label_hamd = np.delete(label_hamd, replicated_indices, axis=0)
demografic_data = np.delete(demografic_data, replicated_indices, axis=0)
baseline_clinical_data = np.delete(baseline_clinical_data, replicated_indices, axis=0)


# baseline HAMD will be added into the baseline_clinical_data 
baseline_clinical_data = np.concatenate((baseline_clinical_data, label_hamd[:, 0:1]), axis=1)


print(f'mdd_subject_base -> {mdd_subject_base.shape}')
print(f'label_hamd -> {label_hamd.shape}')
print(f'demografic_data -> {demografic_data.shape}')
print(f'baseline_clinical_data -> {baseline_clinical_data.shape}')
print('all_involve_subject', all_involve_subject)

# calculate remission 
label_remission = np.zeros(label_hamd.shape[0])
for i, val in enumerate(label_hamd):
    if val[0] >= 7 and val[1] <= 7:
        if val[0] == 7: print(f" val[0] == 7 : i -> {i} | val[1] -> {val[1]}")
        if val[1] == 7: print(f" val[1] == 7 : i -> {i} | val[0] -> {val[0]}")
        label_remission[i] = 1
        # print('label_responder[i] -> ', label_responder[i])
        # print('val -> ',val)
print(label_remission)
count = np.count_nonzero(label_remission == 1)
print(f" number of remission subject in pretreatment -> {count}")


# modify the hb data to be like (subject, 52, 2500)
mdd_subject_base = np.transpose(mdd_subject_base, (0, 2, 1, 3))
hb_data = np.empty((mdd_subject_base.shape[0], 52, 2500))
hbo = mdd_subject_base[:, :, :1250, 0]
hbr = mdd_subject_base[:, :, :1250, 1]
hb_data[:, :, :1250] = hbo
hb_data[:, :, 1250:] = hbr

# genrate adj matrix for gnn_transformer
number_of_subjects = hb_data.shape[0]
adj = generate_fnirs_adj().toarray()
adj = np.tile(adj, (number_of_subjects, 1, 1))
print("adj_matrix shape: ", adj.shape)


output_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/prognosis/pretreatment_remission'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
np.save(output_path + '/hb_data.npy', hb_data)
np.save(output_path + '/label_hamd.npy', label_hamd)
np.save(output_path + '/label_remission.npy', label_remission)
np.save(output_path + '/demografic_data.npy', demografic_data)
np.save(output_path + '/baseline_clinical_data.npy', baseline_clinical_data)
np.save(output_path + '/adj_matrix.npy', adj)