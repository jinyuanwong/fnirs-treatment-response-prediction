import numpy as np 
import sys
import os
import re
def set_path():
    if sys.platform == 'darwin':
        print("Current system is macOS")
        main_fold_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction'
    elif sys.platform == 'linux':
        print("Current system is Ubuntu")
        main_fold_path = '/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning' # '/root/autodl-tmp/fnirs-treatment-response-prediction'
    else:
        print("Current system is neither macOS nor Ubuntu")
    sys.path.append(main_fold_path)
    os.chdir(main_fold_path)
    
set_path()    


import pandas as pd 
def get_best_acc_epoch(csv_file):
    df = pd.read_csv(csv_file)
    index_of_best_acc = df['val_accuracy'].argmax()
    return index_of_best_acc

import glob 

def delete_files_starting_with(directory, prefix):
    pattern = os.path.join(directory, '*' + prefix + '*') 
    files_to_delete = glob.glob(pattern)
    for file in files_to_delete:
        try: 
            os.remove(file)
        except Exception as e:
            print(e)
        
# seeds=[1720351610, 1720353641, 1720355734, 1720358054, 1720360139]
seeds=[31415926, 27182818, 16180339, 12345678, 98765432]

# aug = [0, 1, 2, 3]
# aug = [16, 32, 64, 128, 256, 512]
male_female_arr = ['male', 'female']
# MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_16
dataset = 'diagnosis514'
model_config_dict = {}
# for seed_index, seed in enumerate(seeds):
#     model_config_dict[seed] = [f"MTL_20240710_VF3_AugmentRatio_{seed}MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_{aug[i]}" for i in range(len(aug))]

for index, val in enumerate(male_female_arr):
    model_config_dict[val] = [f"MTL_20240710_VF3_AugmentRatio_{seeds[i]}MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_256_{val}" for i in range(len(seeds))]

total_inner_k = 4
total_outer_k = 5
validation_name = f'nested_cross_validation_outer_{total_outer_k}_inner_{total_inner_k}'

# reading the last index of result
# def find_path_acc(path):
#     with open(path, 'r') as f:
#         acc = f.read()
#     acc = acc.split('\n')[-2]
#     acc = re.findall(r'accuracy: (\d+\.\d+)', acc)[0]
    
#     return np.float(acc)

"""

return accuracy, sensitivity, specificity, auc of the model
"""
def find_path_metrics(path):
    with open(path, 'r') as f:
        content = f.read()

    accuracy = re.findall(r'accuracy: (\d+\.\d+)', content)[0]
    sensitivity = re.findall(r'sensitivity: (\d+\.\d+)', content)[0]
    specificity = re.findall(r'specificity: (\d+\.\d+)', content)[0]
    auc = re.findall(r'AUC: (\d+\.\d+)', content)[0]
    duration = re.findall(r'duration: (\d+\.\d+)', content)[0]
    
    # Convert the extracted strings to floats
    accuracy = float(accuracy)
    sensitivity = float(sensitivity)
    specificity = float(specificity)
    auc = float(auc)
    duration = float(duration)
    
    return accuracy, sensitivity, specificity, auc, duration
from utils.fnirs_utils import print_md_table_val_test_AUC
def read_model_config_result(model_name, config_name, total_outer_k, total_inner_k, validation_name, print_table=True):
    test_metrics = []
    val_metrics = []
    index_best = []
    for outer_k in range(total_outer_k):
        for inner_k in range(total_inner_k):
            path = f'results/{model_name}/{dataset}/{config_name}/{validation_name}/outer_{outer_k}_inner_{inner_k}' 
            fold_test_acc = find_path_metrics(path + f'/test_acc_{metric}.txt')
            fold_val_acc = find_path_metrics(path + f'/val_acc_{metric}.txt')
            # fold_index_best = get_best_acc_epoch(path + f'/history_{metric}.csv')
            # index_best.append(fold_index_best)
            if fold_val_acc[-2] > 0:
                test_metrics.append(fold_test_acc)
                val_metrics.append(fold_val_acc)

            # delete checkpoint files, in case the space is not enough
            delete_files_starting_with(path, 'checkpoint')
    if print_table: print_md_table_val_test_AUC(f"{config_name[-10:]} len({len(val_metrics)})", np.mean(test_metrics, axis=0), np.mean(val_metrics,axis=0), print_table_header=False, already_balanced_accuracy=False)
    # print(val_metrics)
    # print('model_name:', model_name, config_name)
    # print('test_acc:', np.mean(test_metrics))
    # print('val_acc:', np.mean(val_metrics))
    # print('index_best:', np.mean(index_best))
    return np.mean(test_metrics, axis=0), np.mean(val_metrics,axis=0)

all_test_metrics = []
all_val_metrics = []
all_metircs_name = []
from utils.utils_mine import plot_evaluation_metrics_header

metrics={
        'depression': 'accuracy',
        # 'HAMD_Scores': 'accuracy',
        # 'Suicide_Risk': 'accuracy',         
        }
# metrics={
#         'gender': 'accuracy', 
#         'depression': 'accuracy',
#         'HAMD_Scores': 'accuracy',
#         'Suicide_Risk': 'accuracy',         
#         'age': 'accuracy', 
#         'education': 'accuracy', 
#         'smoking': 'accuracy', 
#         'alcohol': 'accuracy',
#         }
# metrics={
#         'STL_degression': 'accuracy', 
#         }
model_name = 'jamba_MTL'
for metric, val in metrics.items():
    all_metircs_name.append(metric)
    print()
    plot_evaluation_metrics_header(table_name = 'Depression', parameter_name='M/F')     
    for val, config_names in model_config_dict.items():
        rep_test_metric = []
        rep_val_metric = []
        for itr_index, config_name in enumerate(config_names):
            test_metrics, val_metrics = read_model_config_result(model_name, config_name, total_inner_k=total_inner_k, total_outer_k=total_outer_k, validation_name=validation_name, print_table=False)
            rep_test_metric.append(test_metrics)
            rep_val_metric.append(val_metrics)
            
        rep_test_metric = np.array(rep_test_metric)
        rep_val_metric = np.array(rep_val_metric)
        print_md_table_val_test_AUC(f"{val}", np.mean(rep_test_metric, axis=0), np.mean(rep_val_metric,axis=0), print_table_header=False, already_balanced_accuracy=False)
        # all_test_metrics.append(rep_test_metric.mean(axis=0))
        # all_val_metrics.append(rep_val_metric.mean(axis=0))
