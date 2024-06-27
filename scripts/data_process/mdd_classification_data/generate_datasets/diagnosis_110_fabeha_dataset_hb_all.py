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
# import scripts.data_process.mdd_classification_data.generate_datasets as generate_datasets
# from generate_datasets import obtain_hb_data_label_hamd
# import scripts.data_process.mdd_classification_data.generate_datasets as generate_datasets, correct_channel, loop_data_path_hb_type

def retrieve_110_index_from_fabeha_according_to_label(label):
    label_0 = np.where(label == 0)[0]
    label_1 = np.where(label == 1)[0]
    print('label_0:', label_0.shape, label_0)
    print('label_1:', label_1.shape, label_1)
    label_0 = label_0[-55:]
    label_1 = label_1[-55:]
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
    hb_data_all, label_all, hamd_all = obtain_hb_data_label_hamd()
    hb_simple_all = get_simple_avg_hb(hb_data_all)
    hb_simple_all_1d = np.concatenate((hb_simple_all[...,0], hb_simple_all[...,1], hb_simple_all[...,2]), axis=-1)

    selected_index = retrieve_110_index_from_fabeha_according_to_label(label_all)
    
    selected_hb, selected_hb_simple_all_1d, selected_label, selected_hamd = select_data(selected_index, hb_data_all, hb_simple_all_1d, label_all, hamd_all)
    
    print('saving hb_data, hb_simple_all_1d, label and hamd with shape:')
    print('selected_label', selected_label)
    print(selected_hb.shape, selected_hb_simple_all_1d.shape, selected_label.shape, selected_hamd.shape)
    
    save_fold = 'allData/diagnosis_110_fabeha_dataset_hb_all/'
    if os.path.exists(save_fold) is False:
        os.makedirs(save_fold)
        print('Create folder:', save_fold)
    else:
        print('Folder exists:', save_fold)
    nor_hb_simple_all_1d = normalize_individual(selected_hb_simple_all_1d)
    np.save(save_fold + 'hb_data.npy', selected_hb)
    np.save(save_fold + 'label.npy', selected_label)
    np.save(save_fold + 'hamd.npy', selected_hamd)
    np.save(save_fold + 'hb_simple_all_1d.npy', selected_hb_simple_all_1d)
    np.save(save_fold + 'nor_hb_simple_all_1d.npy', nor_hb_simple_all_1d)
    
    

    



# save_num_of_subjects = 110

# hb_all = np.concatenate((hbo, hbr, hbt), axis=-1)

# select_index = [i for i in range(save_num_of_subjects)]
# select_index = hb_all.shape[0] - 1 - np.array(select_index)

# selected_data, selected_label = select_data(hb_all, label_all, select_index)
# save_fold = 'allData/diagnosis_110_fabeha_dataset_hb_all/'
# np.save(save_fold + 'hb_data.npy', selected_data)
# np.save(save_fold + 'label.npy', selected_label)