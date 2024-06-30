import numpy as np 
from scipy.io import loadmat
import sys 
import os

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
    sys.path.append(main_fold_path)    

set_path()
from scripts.data_process.mdd_classification_data.generate_datasets import generate_hb_and_label as generate_hb_and_label #as obtain_hb_data_label_hamd
from generate_hb_and_label import obtain_hb_data_label_hamd, get_simple_avg_hb
from generate_hb_and_label import get_nine_regions_hb_mean_task_change
# import scripts.data_process.mdd_classification_data.generate_datasets as generate_datasets
# from generate_datasets import obtain_hb_data_label_hamd
# import scripts.data_process.mdd_classification_data.generate_datasets as generate_datasets, correct_channel, loop_data_path_hb_type

def retrieve_index_from_fabeha_according_to_label(label, number_of_subjects):
    each_label_num = number_of_subjects // 2
    label_0 = np.where(label == 0)[0]
    label_1 = np.where(label == 1)[0]
    print('label_0:', label_0.shape, label_0)
    print('label_1:', label_1.shape, label_1)
    label_0 = label_0[-each_label_num:]
    label_1 = label_1[-each_label_num:]
    selected_index = np.concatenate((label_0, label_1), axis=0)
    return selected_index    
    

def select_data(index, *arrays):
    if index is None or any(arr is None for arr in arrays):
        raise ValueError("One or more inputs are None")
    
    retain_arr = index # [i for i in range(arrays[0].shape[0]) if i not in index]
    return tuple(arr[retain_arr] for arr in arrays)

def show_data(data, save_path='hb_simple_all_1d.png'):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(data.mean(axis=(0,1)))
    plt.savefig(save_path)

from utils.utils_mine import  normalize_individual

        
if __name__ == '__main__':
    
    for number_of_subjects in [110, 250]: # 250 or 110
        hb_data_all, label_all, hamd_all = obtain_hb_data_label_hamd()

        
        
        selected_index = retrieve_index_from_fabeha_according_to_label(label_all, number_of_subjects=number_of_subjects)

        
        selected_hb_data_all, selected_label, selected_hamd = select_data(selected_index, hb_data_all, label_all, hamd_all)
        selected_hb_data_all_1d = np.concatenate((selected_hb_data_all[...,0], selected_hb_data_all[...,1], selected_hb_data_all[...,2]), axis=-1)

        avg_selected_hb_data_all = get_simple_avg_hb(selected_hb_data_all)
        avg_selected_hb_data_all_1d = np.concatenate((avg_selected_hb_data_all[...,0], avg_selected_hb_data_all[...,1], avg_selected_hb_data_all[...,2]), axis=-1)
        
        print('saving hb_data, hb_simple_all_1d, label and hamd with shape:')
        print('selected_label', selected_label)
        print(avg_selected_hb_data_all.shape, avg_selected_hb_data_all_1d.shape, selected_label.shape, selected_hamd.shape)
        
        save_fold = f'allData/diagnosis_{number_of_subjects}_fabeha_dataset_hb_all/'
        if os.path.exists(save_fold) is False:
            os.makedirs(save_fold)
            print('Create folder:', save_fold)
        else:
            print('Folder exists:', save_fold)
        nor_avg_selected_hb_data_all_1d= normalize_individual(avg_selected_hb_data_all_1d)
        np.save(save_fold + 'hb_data_3d.npy', selected_hb_data_all) # subject, channel, timepoint, hbo/hbr/hbt
        np.save(save_fold + 'hb_data_1d.npy', selected_hb_data_all_1d) # subject, channel, timepoint(hbo) + timepoint(hbr) + timepoint(hbt)
        np.save(save_fold + 'nor_hb_data_1d.npy', normalize_individual(selected_hb_data_all_1d)) # subject, channel, timepoint(hbo) + timepoint(hbr) + timepoint(hbt)
        np.save(save_fold + 'nor_seq_ch_hb_data_1d.npy', selected_hb_data_all_1d.transpose(0, 2, 1)) # subject, channel, timepoint(hbo) + timepoint(hbr) + timepoint(hbt)

        np.save(save_fold + 'label.npy', selected_label)
        np.save(save_fold + 'hamd.npy', selected_hamd)
        np.save(save_fold + 'hb_simple_all_1d.npy', avg_selected_hb_data_all_1d)
        np.save(save_fold + 'nor_hb_simple_all_1d.npy', nor_avg_selected_hb_data_all_1d)
        np.save(save_fold + 'nor_seq_ch_hb_simple_all_1d.npy', nor_avg_selected_hb_data_all_1d.transpose(0, 2, 1)) # subject, channel, timepoint(hbo) + timepoint(hbr) + timepoint(hbt)

        nine_regions_hbo_task_change_fnirs_features = get_nine_regions_hb_mean_task_change(selected_hb_data_all, hb_index=0)
        nine_regions_hbr_task_change_fnirs_features = get_nine_regions_hb_mean_task_change(selected_hb_data_all, hb_index=1)
        nine_regions_hbt_task_change_fnirs_features = get_nine_regions_hb_mean_task_change(selected_hb_data_all, hb_index=2)
        np.save(save_fold + 'nine_regions_hbo_task_change_fnirs_features.npy', nine_regions_hbo_task_change_fnirs_features)
        np.save(save_fold + 'nine_regions_hbr_task_change_fnirs_features.npy', nine_regions_hbr_task_change_fnirs_features)
        np.save(save_fold + 'nine_regions_hbt_task_change_fnirs_features.npy', nine_regions_hbt_task_change_fnirs_features)
