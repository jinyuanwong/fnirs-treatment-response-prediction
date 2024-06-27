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

    
def loop_data_path_hb_type(data_pth, Hb_types):
    all_hb_data, label = [], []
    for hb_type in Hb_types:
        path = data_pth + f'/all52CH_prep_{hb_type}.mat'
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

def obtain_hb_data_label_hamd():
    
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

    # Cyrus Febeha
    can_not_find = []

    for i in fabeha_sub_name + cyrus_sub_name:
        if i not in subject_to_hamd_dict:
            # can_not_find.append(i)
            for name_in_hamd in subject_to_hamd_dict:
                if name_in_hamd == i[:len(name_in_hamd)]:
                    print(f"can find {i} | replace with {name_in_hamd} | with score of {subject_to_hamd_dict[name_in_hamd]}")
                    subject_to_hamd_dict[i] = subject_to_hamd_dict[name_in_hamd]
                    break


    for i in fabeha_sub_name + cyrus_sub_name:
        if i not in subject_to_hamd_dict:
            can_not_find.append(i)
    print(can_not_find)       


    # Goal generate a shape including the correct order of label and hamd score 

    delete_subejct_name = 'Cyrus_CT033'
    two_dataset_correct_order_name = cyrus_sub_name + fabeha_sub_name
    delete_index_in_gabrille = np.where(np.array(two_dataset_correct_order_name) == delete_subejct_name)[0][0]
    two_dataset_correct_order_name.remove(delete_subejct_name)

    hb_data_dataset1, label_dataset1 = loop_data_path_hb_type(dataset_1_pth, Hb_types)
    hb_data_dataset2, label_dataset2 = loop_data_path_hb_type(dataset_2_pth, Hb_types)

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

    return hb_data_all, label_all, hamd_all



if __name__ == '__main__':

    hb_data_all, label_all, hamd_all = obtain_hb_data_label_hamd()

    hb_data_all_3d = hb_data_all
    # normalize data by dividing by mean values, and concatenate them together
    # hb_data_all = np.concatenate([hb_data_all[..., i] / np.mean(hb_data_all[..., i]) for i in range(3)], axis=2)
    hb_data_all = np.concatenate([hb_data_all_3d[..., i] for i in range(3)], axis=2)

    # me

    save_fold = 'allData/diagnosis514/'        

    # save data
    np.save(save_fold + 'hb_data.npy', hb_data_all)
    np.save(save_fold + 'label.npy', label_all)



    # average along -1 by every 10pionts 
    hbo = avg_ten_points(hb_data_all_3d[...,:-1,0])

    hbr = avg_ten_points(hb_data_all_3d[...,:-1,1])

    hbt = avg_ten_points(hb_data_all_3d[...,:-1,2])
    np.save(save_fold + 'hbo_simple_data.npy', hbo)

    hb_simple_3d = np.concatenate([hbo[..., np.newaxis], hbr[..., np.newaxis], hbt[..., np.newaxis]], axis=-1)
    np.save(save_fold + 'hb_simple_3d.npy', hb_simple_3d)
    
    hb_simple_all_1d = np.concatenate((hbo, hbr, hbt), axis=-1)
    nor_hb_simple_all_1d = normalize_individual(hb_simple_all_1d)
    np.save(save_fold + 'nor_hb_simple_all_1d.npy', nor_hb_simple_all_1d)
    
    np.save(save_fold + 'hamd.npy', hamd_all)
