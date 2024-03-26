
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import os
import sys 
sys.path.append(os.getcwd())
from utils.fnirs_utils import get_metrics
from utils.fnirs_utils import print_md_table_val_test
from utils.fnirs_utils import convert_result_to_y_pred

import re



def convert_to_float(value):
    value = np.array(value)
    try:
        final_value = value.astype(float)
        np.nan_to_num(final_value, nan=0.0)
    except ValueError:
        final_value = 0.0
    return final_value
# get the cross validation result and the average / max / best_val - test result 
def read_metrics_txt_specify_itr(path, taking_itr):
    with open(path, 'r') as f:
        content = f.read()
        acc = convert_to_float(re.findall(r'accuracy: (\d+\.\d+)', content)[taking_itr])
        sen = convert_to_float(re.findall(r'sensitivity: (\d+\.\d+)', content)[taking_itr])
        spe = convert_to_float(re.findall(r'specificity: (\d+\.\d+)', content)[taking_itr])
        f1 = convert_to_float(re.findall(r'F1-score: (\d+\.\d+)', content)[taking_itr])
    return [acc,sen,spe,f1]


def read_file_metric_acc_sen_spe_f1(path):
    with open(path, 'r') as f:
        content = f.read()
        acc = convert_to_float(re.findall(r'accuracy: (\d+\.\d+)', content))
        sen = convert_to_float(re.findall(r'sensitivity: (\d+\.\d+)', content))
        spe = convert_to_float(re.findall(r'specificity: (\d+\.\d+)', content))
        f1 = convert_to_float(re.findall(r'F1-score: (\d+\.\d+)', content))
    return acc, sen, spe, f1
def read_metrics_txt_best_itr(path, based_best_metric='sensitivity'): # 
    
    acc, sen, spe, f1 = read_file_metric_acc_sen_spe_f1(path)
    # get the itr of the best sensitivity result 
    if based_best_metric == 'sensitivity':
        based_on_best_metric_location = sen # val_f1_itr
    elif based_best_metric == 'f1_score':
        based_on_best_metric_location = f1 # val_f1_itr
    else:
        raise ValueError('based_best_metric should be either sensitivity or f1_score')
    best_index = np.argmax(based_on_best_metric_location)

    # print(f" all sen {sen} - best sen {sen[best_index]} - best index {best_index}")
    res = [acc[best_index],
           sen[best_index],
           spe[best_index],
           f1[best_index]]
    return res, best_index

def get_test_acc_using_val_best_itr(path, itr):
    acc, _, _, _ = read_file_metric_acc_sen_spe_f1(path)
    return acc[itr]
def get_val_metrics_and_test_accuracies(model, val_fold_path, based_best_metric='sensitivity', subject_fold_name='LOO_'):
    # get the averay validation result 
    ## only consider 1 iteration 
    taking_itr = 0
    num_of_cv_folds = len(os.listdir(val_fold_path + '/' + subject_fold_name + str(0)))#3 
    print('val_fold_path', val_fold_path + '/' + subject_fold_name + str(0))
    print('num_of_cv_folds', num_of_cv_folds)
    all_loo_metrics = []
    all_loo_acc = []
    for loo in range(total_subjects):
        ind_loo_folds =[]
        loo_acc = []
        # test_best_itr = []
        for cv_fold in range(num_of_cv_folds):
            # read_fold = f"{val_fold_path}/LOO_{loo}/stratified_nested_{num_of_cv_folds}_CV_fold-{cv_fold}/"
            read_fold = f"{val_fold_path}/LOO_{loo}/stratified_nested_{num_of_cv_folds}_CV_fold-{cv_fold}/"
            read_val_path = read_fold + "val_acc.txt"
            read_test_path = read_fold + "test_acc.txt"

            res_metrics, val_best_itr = read_metrics_txt_best_itr(read_val_path, based_best_metric=based_best_metric)
            all_best_itr_record.append(val_best_itr)
            # test_best_itr.append(val_best_itr)
            cv_fold_acc = get_test_acc_using_val_best_itr(read_test_path, val_best_itr)
            loo_acc.append(cv_fold_acc)
            ind_loo_folds.append(res_metrics)    
        # print('loo_acc', loo_acc, 'best_itr', test_best_itr)
        all_loo_acc.append(np.mean(loo_acc))
        # ind_loo_folds should be a shape like (3, 4) (folds, acc|sen|spe|f1)
        mean_ind_loo_folds = np.mean(ind_loo_folds, axis=0)
        all_loo_metrics.append(mean_ind_loo_folds)
    # all_loo_metrics is a shape like (all_loo, 4->acc\sen\spe\f1)
    mean_all_loo_metrics = np.mean(all_loo_metrics, axis=0)
    # print_md_table(model, 'validation', mean_all_loo_metrics)
    # print('all_loo_acc', all_loo_acc)
    return mean_all_loo_metrics, all_loo_acc



# get the y_pred of model's testing 
def get_y_pred_test(model, val_fold_path):
    # get the averay validation result 
    ## only consider 1 iteration 
    taking_itr = 0
    num_of_cv_folds = 5
    all_loo_metrics = []
    for loo in range(total_subjects):
        ind_loo_folds =[]
        for cv_fold in range(num_of_cv_folds):
            read_path = f"{val_fold_path}/LOO_{loo}/stratified_nested_{num_of_cv_folds}_CV_fold-{cv_fold}/test_acc.txt"
            read_path = f"{val_fold_path}/LOO_{loo}/stratified_nested_CV_fold-{cv_fold}/test_acc.txt"

            with open(read_path, 'r') as f:
                content = f.read()
                acc = convert_to_float(re.findall(r'accuracy: (\d+\.\d+)', content)[taking_itr])
            ind_loo_folds.append(acc)            
        # ind_loo_folds should be a shape like (3, 4) (folds, acc|sen|spe|f1)
        mean_ind_loo_folds = np.mean(ind_loo_folds, axis=0)
        all_loo_metrics.append(mean_ind_loo_folds)
    # all_loo_metrics is a shape like (all_loo, 4->acc\sen\spe\f1)
    # mean_all_loo_metrics = np.mean(all_loo_metrics, axis=0)
    # print_md_table(model, 'validation', mean_all_loo_metrics)
    print('all_loo_metrics', all_loo_metrics)
    return np.array(all_loo_metrics)


# model = 'yu_gnn'  # comb_cnn or cnn_transformer gnn_transformer gnn yu_gnn
# model_params = 'v1' # 'd_model_16_batch_size_64_n_layers_6'  # V1
model = 'gnn_transformer'  # comb_cnn or cnn_transformer gnn_transformer gnn yu_gnn
model_params = 'v2l1_rate_0.01_l2_rate_0.01_d_model_16_batch_size_64_n_layers_6' # 'd_model_16_batch_size_64_n_layers_6'  # V1


time = 'prognosis/pre_treatment_hamd_reduction_50'
# 'pre_treatment_hamd_reduction_50' or 'pre_post_treatment_hamd_reduction_50'

validation_method = 'LOO_nested_CV'  # 'LOOCV' or 'k_fold' LOO_nested_CV
based_best_metric = 'sensitivity' # 'sensitivity' or 'f1_score'
all_best_itr_record = []


val_fold_path = f'results/{model}/{time}/{model_params}/LOO_nested_CV'

output_fold = f'FigureTable/DL/timedomain/{time}'

if not os.path.exists(output_fold):
    os.makedirs(output_fold)

# y_test_path = f'allData/prognosis/{time}'
y_test_path = f'allData/{time}/{model}'
y_test_path = f'allData/prognosis/pre_treatment_hamd_reduction_50'

total_subjects  = 46 if time[:8] == 'pre_post' else 65 # '65' or '45

val_nested_CV_metrics, test_accuracy = get_val_metrics_and_test_accuracies(model, val_fold_path, based_best_metric=based_best_metric)


# test_accuracy = get_y_pred_test(model, val_fold_path)
y_test = np.load(y_test_path + '/label.npy')
y_pred = convert_result_to_y_pred(test_accuracy, y_test)
test_metrics = get_metrics(y_test, y_pred)


print_md_table_val_test(model, test_metrics, val_nested_CV_metrics)
print()
print(all_best_itr_record)