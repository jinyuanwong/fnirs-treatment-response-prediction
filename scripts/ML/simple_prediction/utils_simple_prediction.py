import numpy as np 
import os
import sys 
def set_path():
    if sys.platform == 'darwin':
        print("Current system is macOS")
        main_fold_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction'
    elif sys.platform == 'linux':
        print("Current system is Ubuntu")
        main_fold_path = '/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning'
    else:
        print("Current system is neither macOS nor Ubuntu")
    os.chdir(main_fold_path)
    
def load_data_for_classification():
    data_pth = 'allData/prognosis_mix_hb/pretreatment_response/nor_demo_cli_merge_data.npy'
    label_pth = 'allData/prognosis_mix_hb/pretreatment_response/label.npy'
    data = np.load(data_pth)
    labels = np.load(label_pth)
    return data, labels

def add_cgi(data):
    cgi_path = 'allData/prognosis_mix_hb/pretreatment_response/nor_T2_SDS_CGI.npy'
    cgi = np.load(cgi_path)
    data = np.concatenate((data, cgi[:, :6]), axis=1)
    return data

def add_mddr(data):
    mddr_path = 'allData/prognosis_mix_hb/pretreatment_response/MDDR/MDDR_derived_from_load_evaluate.npy'
    mddr = np.load(mddr_path)
    mddr = mddr[..., -1]
    mddr = np.transpose(mddr, (1, 0))
    data = np.concatenate((data, mddr), axis=1)
    return data


def load_task_change_data():
    data_pth = 'allData/prognosis_mix_hb/pretreatment_response/task_change_hb.npy'
    data = np.load(data_pth)
    return data

def add_task_change_data(data, index=2):
    """
    index: 0 - HbO, 1 - HbR, 2 - HbT
    """
    task_change_data = load_task_change_data()[..., index]
    data = np.concatenate((data, task_change_data), axis=1)
    return data

def load_data():
    data_pth = 'allData/prognosis_mix_hb/pretreatment_response/merge_psychiatry_demo_dose_data.npy'
    label_pth = 'allData/prognosis_mix_hb/pretreatment_response/label_hamd.npy'
    data = np.load(data_pth)
    labels = np.load(label_pth)
    labels = change_hamd_to_percent(labels)
    return data, labels