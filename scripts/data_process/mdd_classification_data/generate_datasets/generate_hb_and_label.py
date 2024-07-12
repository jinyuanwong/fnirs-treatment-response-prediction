import numpy as np 
from scipy.io import loadmat
import sys 
import os
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

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
from utils.utils_mine import normalize_individual

# test one 

# data_pth = 'Prerequisite/data_all_original/Fabeha_s Data/All 52-channel/all52CH_prep_HbO.mat'
# data = loadmat(data_pth)
def avg_ten_points(hb):
    hb = np.mean(hb.reshape(hb.shape[0], hb.shape[1], hb.shape[2]//10, 10), axis=-1)
    return hb
def read_hb_data_label(data): 
    hb_data = data.get('data_hb')
    label = data.get('label')
    return hb_data, label

    
def loop_data_path_hb_type(data_pth, Hb_types, datatype='prep'):
    all_hb_data, label = [], []
    for hb_type in Hb_types:
        path = data_pth + f'/all52CH_{datatype}_{hb_type}.mat'
        print(path)
        data = loadmat(path)
        hb_data, label = read_hb_data_label(data)
        all_hb_data.append(hb_data)
        label = label
    return all_hb_data, label
        


# correct channel 
def correct_channel(data):
    map_location = 'allData/diagnosis/Channel_Configuration.mat'
    channel_map = loadmat(map_location)
    all_channel = [ch[0] for ch in channel_map['Channel'][0][0][0][0]] # ['C29', 'C39', 'C18', 'C28', 'C30', 'C31', 'C41', 'C20', 'C43', 'C32', 'C44', 'C45', 'C34', 'C46', 'C47', 'C36', 'C48', 'C49', 'C38', 'C40', 'C50', 'C51', 'C42', 'C52', 'C1', 'C11', 'C2', 'C3', 'C13', 'C4', 'C5', 'C15', 'C6', 'C7', 'C17', 'C19', 'C8', 'C9', 'C21', 'C10', 'C12', 'C33', 'C22', 'C23', 'C35', 'C14', 'C24', 'C25', 'C37', 'C16', 'C26', 'C27']
    correct_location = [all_channel.index('C'+str(i)) for i in range(1, 53)] # [index_of_'C1', index_of_'C2', index_of_'C3', ...]
    data = data[:, correct_location, ...]
    return data

def get_simple_avg_hb(data):

    hbo = avg_ten_points(data[...,:-1,0])
    hbr = avg_ten_points(data[...,:-1,1])
    hbt = avg_ten_points(data[...,:-1,2])
    
    hb_simple = np.concatenate([hbo[..., np.newaxis], hbr[..., np.newaxis], hbt[..., np.newaxis]], axis=-1)
    
    return hb_simple


from utils.fnirs_utils import get_nine_region_data
from scipy.stats import zscore

def get_nine_regions_hb_mean_task_change(hb_data_all, hb_index):
    """ 
    Parameters: 
        hb_data_all: shape (514 - subject, 9 - region/channel, 1251 - timepoint, 3 - HbO/HbR/HbT)
        hb_index: 0 - HbO, 1 - HbR, 2 - HbT
    
    outputs: 
    res: shape (514, 9, 2) -> (514, 18) -> normalize -> (514, 18)
    """
    
    HB = get_nine_region_data(hb_data_all[..., hb_index])
    task_start_index = 100
    task_end_index = 700
    
    shape = HB.shape[0], HB.shape[1], 2            
    res = np.zeros(shape)
    for i in range(HB.shape[1]):    
        task_change_feature = HB[:, i, task_end_index:].mean(axis=1) - HB[:, i, :task_start_index].mean(axis=1)
        mean_feature = HB[:, i, :].mean(axis=1) 
        res[:, i, 0] = task_change_feature
        res[:, i, 1] = mean_feature
    
    res = res.reshape(res.shape[0], -1)
    normalized_res = zscore(res)    
    
    return normalized_res

def calculate_slope(y, start_idx, end_idx):
    """
    Calculate the slope of the linear fit for the data points between start_idx and end_idx.
    
    Parameters:
    - y: numpy array of data points
    - start_idx: starting index
    - end_idx: ending index
    
    Returns:
    - slope: slope of the linear fit
    """
    x = np.arange(start_idx, end_idx)
    y = y[start_idx:end_idx]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m

def extract_features(data):
    """
    Extract mean for five periods and slope for four periods.
    
    Parameters:
    - data: numpy array of shape (subjects, channels, time_steps)
    
    Returns:
    - features: numpy array of shape (subjects, channels, features)
    """
    
    n_subjects, n_channels, n_time_steps = data.shape
    
    mean_features = np.zeros((n_subjects, n_channels, 5))
    slope_features = np.zeros((n_subjects, n_channels, 4))
    
    mean_periods = [
        (0, 100),
        (100, 300),
        (300, 500),
        (500, 700),
        (700, n_time_steps)
    ]
    
    slope_periods = [
        (50, 150),
        (250, 350),
        (450, 550),
        (650, 750)
    ]
    
    for subject in range(n_subjects):
        for channel in range(n_channels):
            # Compute means for each of the five periods
            for i, (start_idx, end_idx) in enumerate(mean_periods):
                mean_features[subject, channel, i] = np.mean(data[subject, channel, start_idx:end_idx])
            
            # Compute slopes for each of the four periods using the slope function
            for i, (start_idx, end_idx) in enumerate(slope_periods):
                slope_features[subject, channel, i] = calculate_slope(data[subject, channel], start_idx, end_idx)
    
    # Concatenate mean and slope features
    features = np.concatenate((mean_features, slope_features), axis=2)
    return features

def extract_combined_features(hb_data_all):
    """
    Extract combined features for all Hb types and normalize the result.
    
    Parameters:
    - hb_data_all: **unnormalized** numpy array of shape (subjects, channels, time_steps, hb_type)
    
    Returns:
    - combined_features: numpy array of shape (subjects, channels, features*3)
    """
    
    n_subjects, n_channels, n_time_steps, n_hb_types = hb_data_all.shape
    combined_features = []
    
    for hb_index in range(n_hb_types):
        hb_features = extract_features(hb_data_all[..., hb_index])
        combined_features.append(hb_features)
    
    # Combine features for all Hb types
    combined_features = np.concatenate(combined_features, axis=2)

    # # Normalize by dividing by the mean across subjects and channels
    # feature_means = combined_features.mean(axis=(0, 1))
    # normalized_features = combined_features / feature_means
                
    return combined_features

def concatenate_hb_data_and_features(hb_data_all, normalized_features):
    """
    Extract combined features for all Hb types and normalize the result.
    
    Parameters:
    - hb_data_all: **unnormalized** numpy array of shape (subjects, channels, time_steps, hb_type)
    - features: numpy array of shape (subjects, channels, features)
    
    Returns:
    - concatenated_features: numpy array of shape (subjects, channels, features*3)
    """    
    n_subjects, n_channels, n_time_steps, n_hb_types = hb_data_all.shape

    # process input data to concatenate it from 4d -> 3d and normalize it using mean(0,1)
    hb_data_all_1d = np.concatenate([hb_data_all[..., i] for i in range(3)], axis=2)
    normalized_hb_data_all_1d = hb_data_all_1d / hb_data_all_1d.mean(axis=(0, 1))
    
    # Add zeros to reach the desired feature size of 52
    zeros_to_add = normalized_hb_data_all_1d.shape[2] - normalized_features.shape[2]
    if zeros_to_add > 0:
        padding = np.zeros((n_subjects, n_channels, zeros_to_add))
        normalized_features = np.concatenate((normalized_features, padding), axis=2)
        
    concatenated_features = np.concatenate((normalized_hb_data_all_1d, normalized_features), axis=1)    
    
    return concatenated_features

def obtain_hb_data_label_hamd(datatype='prep'):
    
    raw_multi_task_label = pd.read_csv('Prerequisite/patients_info/subject_table.csv')
    multi_task_label = []
    data_fold = 'Prerequisite/data_all_original'

    dataset_1_pth = data_fold + '/Gabrielle_s Data/All 52-channel/'        
    dataset_2_pth = data_fold + '/Fabeha_s Data/All 52-channel/'       
    
    Hb_types = ['HbO', 'HbR', 'HbT']

    # For each dataset, raw/prep HbO/HbR/HbT have the same subject name including order
    cyrus_sub_name = loadmat(dataset_1_pth + 'all52CH_raw_HbO.mat')['subject_name']
    fabeha_sub_name = loadmat(dataset_2_pth + 'all52CH_raw_HbO.mat')['subject_name']
    fabeha_sub_name = [ 'Fabeha_' + i[0] for i in fabeha_sub_name[0, :]]
    cyrus_sub_name = [ 'Cyrus_' + i[0][:-4] for i in cyrus_sub_name[0, :]]


    # read HAM-D score
    hamd_path = 'Prerequisite/HAM-D Scores.mat'
    hamd = loadmat(hamd_path)
    subject_id_names = [ i[0] for i in hamd['subject_ids'][:, 0]]
    hamd = [ i for i in hamd['hamd_score'][:, 0]]

    subject_to_hamd_dict = {}
    for i in range(len(hamd)):
        subject_to_hamd_dict[subject_id_names[i]] = hamd[i]
        
    print(subject_to_hamd_dict)    
    # data['subject_name']

    # Goal generate a shape including the correct order of label and hamd score 

    delete_subejct_name = 'Cyrus_CT033'
    two_dataset_correct_order_name = cyrus_sub_name + fabeha_sub_name
    delete_index_in_gabrille = np.where(np.array(two_dataset_correct_order_name) == delete_subejct_name)[0][0]
    two_dataset_correct_order_name.remove(delete_subejct_name)
    
    # Cyrus Febeha
    can_not_find = []
    print('fabeha_sub_name + cyrus_sub_name', fabeha_sub_name + cyrus_sub_name)
    for name_index, i in enumerate(two_dataset_correct_order_name):
        if i not in subject_to_hamd_dict:
            # can_not_find.append(i)
            for name_in_hamd in subject_to_hamd_dict:
                if name_in_hamd == i[:len(name_in_hamd)]:
                    print(f"can find {i} | replace with {name_in_hamd} | with score of {subject_to_hamd_dict[name_in_hamd]}")
                    subject_to_hamd_dict[i] = subject_to_hamd_dict[name_in_hamd]
                    two_dataset_correct_order_name[name_index] = name_in_hamd
                    break
        print(i)
        multi_task_label.append(raw_multi_task_label[raw_multi_task_label['ID'] == two_dataset_correct_order_name[name_index]].values[0][1:])
    multi_task_label = np.array(multi_task_label)
    for i in two_dataset_correct_order_name:
        if i not in subject_to_hamd_dict:
            can_not_find.append(i)
    print(can_not_find)       




    hb_data_dataset1, label_dataset1 = loop_data_path_hb_type(dataset_1_pth, Hb_types, datatype)
    hb_data_dataset2, label_dataset2 = loop_data_path_hb_type(dataset_2_pth, Hb_types, datatype)

    hb_data_dataset1 = np.array(hb_data_dataset1)
    hb_data_dataset2 = np.array(hb_data_dataset2)
    hb_data_all = np.concatenate((hb_data_dataset1, hb_data_dataset2), axis=1)
    hb_data_all = np.transpose(hb_data_all, (1,2,3,0))

    label_dataset1 = np.array(label_dataset1)
    label_dataset2 = np.array(label_dataset2)
    label_all = np.concatenate((label_dataset1, label_dataset2), axis=0)[:, 0]


    # delete data with index 32 in gabreille
    # Gabrielle - CT-033 == 32 -index delete it [to answer why delete it?]
    # Fabeha has 374, Gabrielle has 141, fabrielle will be the first 
    hb_data_all = np.delete(hb_data_all, delete_index_in_gabrille, axis=0)
    label_all = np.delete(label_all, delete_index_in_gabrille, axis=0)
    hb_data_all = correct_channel(hb_data_all)
    
    # correct order of hamd_score 
    hamd_all = []
    for i, subject_name in enumerate(two_dataset_correct_order_name):
        sub_label = label_all[i]
        sub_hamd = subject_to_hamd_dict[subject_name]
        hamd_all.append(sub_hamd)
    hamd_all = np.array(hamd_all)

    return hb_data_all, label_all, hamd_all, multi_task_label

"""
Args:
    labels: (subject, num_of_labels=8)
        -Correct_Order 'Gender', 'Age', 'Education', 'Smoking', 'Alcohol', 'HAMD_Scores', 'Suicide_Risk', 'Depression'
    
    specify_return_labels (default: 'all')
        - 'all' ('Gender', 'Age', 'Education', 'Smoking', 'Alcohol', 'HAMD_Scores', 'Suicide_Risk', 'Depression')
        - 'three_depression_metrics' (HAMD_Scores, Suicide_Risk, Depression)
        - 'gender_three_depression_metrics' (Gender, HAMD_Scores, Suicide_Risk, Depression)
        - 'gender_age_three_depression_metrics' (Gender, Age, HAMD_Scores, Suicide_Risk, Depression)
    
    
output:
    transpose_labels: (subject, num_of_labels, 2)

"""
def one_hot_encode_labels_for_multitask_learning_and_save(labels, save_fold, specify_return_labels='all'):
    print('labels.shape', labels.shape)
    if specify_return_labels == 'all':
        return_index = np.arange(labels.shape[1])
    elif specify_return_labels == 'three_depression_metrics':
        return_index = [5, 6, 7]  # Indices for 'HAMD_Scores', 'Suicide_Risk', 'Depression'
    elif specify_return_labels == 'gender_three_depression_metrics':
        return_index = [0, 5, 6, 7]  # Indices for 'Gender', 'HAMD_Scores', 'Suicide_Risk', 'Depression'
    elif specify_return_labels == 'gender_age_three_depression_metrics':
        return_index = [0, 1, 5, 6, 7]  # Indices for 'Gender', 'Age', 'HAMD_Scores', 'Suicide_Risk', 'Depression'
    elif specify_return_labels == 'gender':
        return_index = [0]
    elif specify_return_labels == 'depression':
        return_index = [7]        
    else:
        raise Exception("Invalid specify_return_labels value")
        
    labels = labels[:, return_index]
    encoded_labels = []
    for index, label in enumerate(labels.T):
        label = label.reshape(-1, 1)
        # print(label, label.shape)
        encoded = OneHotEncoder().fit_transform(label)
        encode = encoded.toarray()[..., np.newaxis]
        encoded_labels.append(encode)
        
    concate_labels = np.concatenate(encoded_labels, axis=-1)
    # from shape (subject, 2, num_of_labels) to (subject, num_of_labels, 2)
    transpose_labels = np.transpose(concate_labels, (0, 2, 1))
    
    np.save(save_fold + f'multi_task_label_{specify_return_labels}_onehot.npy', transpose_labels)

    return transpose_labels


if __name__ == '__main__':
    hb_data_all, label_all, hamd_all, multi_task_label = obtain_hb_data_label_hamd()
    hb_data_all_3d = hb_data_all

    
    # normalize data by dividing by mean values, and concatenate them together
    # hb_data_all = np.concatenate([hb_data_all[..., i] / np.mean(hb_data_all[..., i]) for i in range(3)], axis=2)
    hb_data_all_1d = np.concatenate([hb_data_all_3d[..., i] for i in range(3)], axis=2)

    save_fold = 'allData/diagnosis514/'        

    # average along -1 by every 10pionts 
    hbo = avg_ten_points(hb_data_all_3d[...,:-1,0])
    hbr = avg_ten_points(hb_data_all_3d[...,:-1,1])
    hbt = avg_ten_points(hb_data_all_3d[...,:-1,2])
    hb_simple_3d = np.concatenate([hbo[..., np.newaxis], hbr[..., np.newaxis], hbt[..., np.newaxis]], axis=-1)
    hb_simple_all_1d = np.concatenate((hbo, hbr, hbt), axis=-1)
    
    
    nor_hb_data_all_1d = normalize_individual(hb_data_all_1d)
    nor_hb_simple_all_1d = normalize_individual(hb_simple_all_1d)

    np.save(save_fold + 'hb_data_3d.npy', hb_data_all_3d) # subject, channel, timepoint, hbo/hbr/hbt
    np.save(save_fold + 'hb_data_1d.npy', hb_data_all_1d) # subject, channel, timepoint(hbo) + timepoint(hbr) + timepoint(hbt)

    np.save(save_fold + 'nor_hb_data_1d.npy', normalize_individual(hb_data_all_1d)) # subject, channel, timepoint(hbo) + timepoint(hbr) + timepoint(hbt)
    np.save(save_fold + 'nor_seq_ch_hb_data_1d.npy', nor_hb_data_all_1d.transpose(0, 2, 1)) # subject, channel, timepoint(hbo) + timepoint(hbr) + timepoint(hbt)
    np.save(save_fold + 'label.npy', label_all)
    np.save(save_fold + 'hbo_simple_data.npy', hbo)
    np.save(save_fold + 'hb_simple_3d.npy', hb_simple_3d)    
    np.save(save_fold + 'nor_hb_simple_all_1d.npy', nor_hb_simple_all_1d)
    np.save(save_fold + 'nor_seq_ch_hb_simple_all_1d.npy', nor_hb_simple_all_1d.transpose(0, 2, 1)) # subject, channel, timepoint(hbo) + timepoint(hbr) + timepoint(hbt)
    np.save(save_fold + 'hamd.npy', hamd_all)

    extracted_featues = extract_combined_features(hb_data_all_3d)
    np.save(save_fold + 'extracted_featues.npy', extracted_featues.reshape(extracted_featues.shape[0], -1))

    nor_hb_simple_all_1d_conc_features = concatenate_hb_data_and_features(hb_simple_3d, extracted_featues)    
    np.save(save_fold + 'nor_hb_simple_all_1d_conc_features.npy', nor_hb_simple_all_1d_conc_features)

    
    nine_regions_hbo_task_change_fnirs_features = get_nine_regions_hb_mean_task_change(hb_data_all_3d, hb_index=0)
    nine_regions_hbr_task_change_fnirs_features = get_nine_regions_hb_mean_task_change(hb_data_all_3d, hb_index=1)
    nine_regions_hbt_task_change_fnirs_features = get_nine_regions_hb_mean_task_change(hb_data_all_3d, hb_index=2)
    np.save(save_fold + 'nine_regions_hbo_task_change_fnirs_features.npy', nine_regions_hbo_task_change_fnirs_features)
    np.save(save_fold + 'nine_regions_hbr_task_change_fnirs_features.npy', nine_regions_hbr_task_change_fnirs_features)
    np.save(save_fold + 'nine_regions_hbt_task_change_fnirs_features.npy', nine_regions_hbt_task_change_fnirs_features)

    np.save(save_fold + 'multi_task_label.npy', multi_task_label)
    
    multi_task_label_all_onehot = one_hot_encode_labels_for_multitask_learning_and_save(multi_task_label, save_fold, specify_return_labels='all')
    multi_task_label_three_depression_metrics_onehot = one_hot_encode_labels_for_multitask_learning_and_save(multi_task_label, save_fold, specify_return_labels='three_depression_metrics')
    multi_task_label_gender_three_depression_metrics_onehot = one_hot_encode_labels_for_multitask_learning_and_save(multi_task_label, save_fold, specify_return_labels='gender_three_depression_metrics')
    multi_task_label_gender_age_three_depression_metrics_onehot = one_hot_encode_labels_for_multitask_learning_and_save(multi_task_label, save_fold, specify_return_labels='gender_age_three_depression_metrics')
    multi_task_label_gender_age_three_depression_metrics_onehot = one_hot_encode_labels_for_multitask_learning_and_save(multi_task_label, save_fold, specify_return_labels='gender')
    task_label_depression_onehot = one_hot_encode_labels_for_multitask_learning_and_save(multi_task_label, save_fold, specify_return_labels='depression')
    
    # generate female/male - depression classification 
    male_index = multi_task_label[:, 0] == 0
    female_index = multi_task_label[:, 0] == 1 
    
    male_hb_data = nor_hb_simple_all_1d[male_index]
    female_hb_data = nor_hb_simple_all_1d[female_index]
    male_multi_task_label_depression_onehot = task_label_depression_onehot[male_index]
    female_multi_task_label_depression_onehot = task_label_depression_onehot[female_index]
    np.save(save_fold + 'male_nor_hb_simple_all_1d.npy', male_hb_data)
    np.save(save_fold + 'female_nor_hb_simple_all_1d.npy', female_hb_data)
    np.save(save_fold + 'male_multi_task_label_depression_onehot.npy', male_multi_task_label_depression_onehot)
    np.save(save_fold + 'female_multi_task_label_depression_onehot.npy', female_multi_task_label_depression_onehot)
    
    