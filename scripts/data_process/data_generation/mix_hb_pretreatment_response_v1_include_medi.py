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
from utils.fnirs_utils import read_demographic, read_clinical_history, read_psychiatry_history, read_HAMD_ALL_HISTORY

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


excel_data = pd.read_excel(cli_path, sheet_name='Summary T0T8_fNIRS Analysis')
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

medi_dose_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/cyrus_follow-up/data-information/medicine_info.xlsx'
medi_dose_data = pd.read_excel(medi_dose_path, sheet_name='Sheet1')
dose_information = []
count_not_in_medi_dose = 0

empty_subject_dose = [np.nan]*7



mdd_subject_base = []#np.zeros((len(all_subject), 1251, 52, 2)) # time, channel, hbo/hbr
all_involve_subject = []
for sub_index, subject in enumerate(all_subject):
    
    hamd_of_id_t1 = excel_data[excel_data['Subject ID'] == subject]['HAM-D Questionnaire (T1)'].iloc[0]
    hamd_of_id_t8 = excel_data[excel_data['Subject ID'] == subject]['HAM-D Questionnaire (T8)'].iloc[0]
    if type(hamd_of_id_t8) is not int:
        print(hamd_of_id_t8)
        continue
    sub_label = [hamd_of_id_t1, hamd_of_id_t8]
    label_hamd.append(sub_label)
    
    demographic = excel_data[excel_data['Subject ID'] == subject].iloc[:, 2:13]
    demografic_data.append(demographic)
    
    clinical = cgi_sgs_data[cgi_sgs_data['Subject ID'] == subject].iloc[:, 1:7]
    baseline_clinical_data.append(clinical)
    
    medi_dose_subjects = medi_dose_data['Subject ID'].values
    if subject not in medi_dose_subjects:
        dose_information.append(empty_subject_dose)
    else:
        sub_dose_info = medi_dose_data[medi_dose_data['Subject ID'] == subject].iloc[:,3:10].values.tolist()[0]
        dose_information.append(sub_dose_info)

    
    all_involve_subject.append(subject)
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
dose_information = np.array(dose_information)


# check if there is any replicated subject, becasue there might be two files with same subject names
replicated_indices = check_replicate_subject(all_subject)
print(f'return replicated_indices {replicated_indices}')
all_involve_subject = np.array(all_involve_subject)
all_involve_subject = np.delete(all_involve_subject, replicated_indices, axis=0)

HAMD_ALL_HISTORY = read_HAMD_ALL_HISTORY(cli_path, all_involve_subject)
PSYCHIATRY_HISTORY = read_psychiatry_history(cli_path, all_involve_subject)
CLINICAL_HISTORY = read_clinical_history(cli_path, all_involve_subject)
demographic = read_demographic(cli_path, all_involve_subject)


print('HAMD_ALL_HISTORY.shape', HAMD_ALL_HISTORY.shape)
print('PSYCHIATRY_HISTORY.shape', PSYCHIATRY_HISTORY.shape)
print('CLINICAL_HISTORY.shape', CLINICAL_HISTORY.shape)
print('demographic', demographic.shape)



# delete the replicated subject
mdd_subject_base = np.delete(mdd_subject_base, replicated_indices, axis=0)
label_hamd = np.delete(label_hamd, replicated_indices, axis=0)
demografic_data = np.delete(demografic_data, replicated_indices, axis=0)
baseline_clinical_data = np.delete(baseline_clinical_data, replicated_indices, axis=0)
dose_information = np.delete(dose_information, replicated_indices, axis=0)

# baseline HAMD will be added into the baseline_clinical_data 
baseline_clinical_data = np.concatenate((baseline_clinical_data, label_hamd[:, 0:1]), axis=1)


print(f'mdd_subject_base -> {mdd_subject_base.shape}')
print(f'label_hamd -> {label_hamd.shape}')
print(f'demografic_data -> {demografic_data.shape}')
print(f'baseline_clinical_data -> {baseline_clinical_data.shape}')
print(f'dose_information -> {dose_information.shape}')
print('all_involve_subject', all_involve_subject)







# calculate remission 
label_response = np.zeros(label_hamd.shape[0])
for i, val in enumerate(label_hamd):
    if (val[1] - val[0]) / val[0] <= -0.5:
        label_response[i] = 1
        # print('label_responder[i] -> ', label_responder[i])
        # print('val -> ',val)
print(label_response)
count = np.count_nonzero(label_response == 1)
print(f" number of remission subject in pretreatment -> {count}")


# modify the hb data to be like (subject, 52, 2500)
mdd_subject_base = mdd_subject_base[:, :1250, :, :]
mdd_subject_base = mdd_subject_base.transpose((0, 2, 1, 3))
mdd_subject_base = mdd_subject_base.reshape((mdd_subject_base.shape[0], 52, -1))

hb_data = mdd_subject_base

# genrate adj matrix for gnn_transformer
number_of_subjects = hb_data.shape[0]
adj = generate_fnirs_adj().toarray()
adj = np.tile(adj, (number_of_subjects, 1, 1))
print("adj_matrix shape: ", adj.shape)


output_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/prognosis_mix_hb/pretreatment_response'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
np.save(output_path + '/hb_data.npy', hb_data)
np.save(output_path + '/label_hamd.npy', label_hamd)
np.save(output_path + '/label_response.npy', label_response)
np.save(output_path + '/label.npy', label_response)
np.save(output_path + '/demografic_data.npy', demografic_data) # demografic is 2-13 (2-9 is demographic, 10-13 is clinical)
np.save(output_path + '/baseline_clinical_data.npy', baseline_clinical_data)
np.save(output_path + '/adj_matrix.npy', adj)

np.save(output_path + '/HAMD_ALL_HISTORY.npy', HAMD_ALL_HISTORY)
np.save(output_path + '/PSYCHIATRY_HISTORY.npy', PSYCHIATRY_HISTORY)
np.save(output_path + '/CLINICAL_HISTORY.npy', CLINICAL_HISTORY)
np.save(output_path + '/demographic.npy', demographic)
np.save(output_path + '/dose_information.npy', dose_information)
