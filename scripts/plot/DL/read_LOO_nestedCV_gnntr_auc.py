# python scripts/plot/DL/read_LOO_nestedCV_gnntr.py --model gnn_transformer --max 5 --dataset posttreatment_response 
dict_model_params = {
    'gnn_transformer': 'loocv_v1l1_rate_0.01_l2_rate_0.01_d_model_16_batch_size_64_n_layers_6', # 'v2_repeat_3l1_rate_0.01_l2_rate_0.01_d_model_16_batch_size_64_n_layers_6',#
    'gnn_transformer_tp_fc_fs': 'v1l1_rate_0.01_l2_rate_0.01_d_model_16_batch_size_64_n_layers_6',
    'gnn_transformer_tp_dp': 'v1l1_rate_0.01_l2_rate_0.01_d_model_16_batch_size_64_n_layers_6',
    'decisiontree': 'v1',
    'zhu_xgboost': 'v1',
    'fusion_xgboost': 'loocv_v2',
    'fusion_catboost': 'testiterations_1000_learning_rate_0.1_depth_6',
    'wang_alex': 'v1lr_0.001_activation_relu',
    'yu_gnn': 'v1'
}
import numpy as np
import matplotlib.pyplot as plt
import os
import sys 
sys.path.append(os.getcwd())
from utils.fnirs_utils import get_metrics, get_metrics_auc
from utils.fnirs_utils import print_md_table_val_test
from utils.fnirs_utils import save_itr_version_with_test_val_auc
from utils.fnirs_utils import convert_result_to_y_pred
import re
from sklearn.metrics import roc_auc_score
import argparse

def print_result_detail_in_every_fold(ALL_BEST_ITR, ALL_TOTAL_ITERATION, predict_accuracy_flag, y_test):
    for i, v in enumerate(ALL_TOTAL_ITERATION):
        loo = i // 5 
        v = i % 5 
        print(f"LOO_{loo} - fold_{v} - Best itr: {ALL_BEST_ITR[i]} - Total itr: {ALL_TOTAL_ITERATION[i]} - acc: {predict_accuracy_flag[loo]}, y_test: {y_test[loo]}")

def convert_to_float(value):
    value = np.array(value)
    try:
        final_value = value.astype(float)
        np.nan_to_num(final_value, nan=0.0)
    except ValueError:
        final_value = 0.0
    return final_value

def read_file_metric_acc_sen_spe_f1(path):
    with open(path, 'r') as f:
        content = f.read()
        acc = convert_to_float(re.findall(r'accuracy: (\d+\.\d+)', content))
        sen = convert_to_float(re.findall(r'sensitivity: (\d+\.\d+)', content))
        spe = convert_to_float(re.findall(r'specificity: (\d+\.\d+)', content))
        f1 = convert_to_float(re.findall(r'F1-score: (\d+\.\d+)', content))
    return acc, sen, spe, f1

def read_file_metric_acc_sen_spe_auc(path):
    with open(path, 'r') as f:
        content = f.read()
        acc = convert_to_float(re.findall(r'accuracy: (\d+\.\d+)', content))
        sen = convert_to_float(re.findall(r'sensitivity: (\d+\.\d+)', content))
        spe = convert_to_float(re.findall(r'specificity: (\d+\.\d+)', content))
        auc = convert_to_float(re.findall(r'AUC: (\d+\.\d+)', content))
    return acc, sen, spe, auc

def read_file_metric_y_pred(path):

    pattern = r"Y_pred_in_test: \[(.*?)\]"
    with open(path, 'r') as f:
        content = f.read()
        y_pred = re.findall(pattern, content)
    numbers_list = np.array([float(num) for s in y_pred for num in s.split()]).reshape(-1, 2).tolist()
    return numbers_list
def read_metrics_txt_best_itr(path, MAX_ITR, based_best_metric='sensitivity'): # 
    
    acc, sen, spe, auc = read_file_metric_acc_sen_spe_auc(path)
    _, _, _, f1 = read_file_metric_acc_sen_spe_f1(path)
    y_pred = read_file_metric_y_pred(path)
    # get the itr of the best sensitivity result 
    if based_best_metric == 'sensitivity':
        based_on_best_metric_location = sen # val_f1_itr
    elif based_best_metric == 'f1_score':
        based_on_best_metric_location = f1 # val_f1_itr
    else:
        raise ValueError('based_best_metric should be either sensitivity or f1_score')
    based_on_best_metric_location = based_on_best_metric_location[:MAX_ITR]
    best_index = np.argmax(based_on_best_metric_location)
    if specify_itr is not None: best_index = specify_itr
    # best_index = np.min([best_index, MAX_ITR])
    # print(f" all sen {sen} - best sen {sen[best_index]} - best index {best_index}")
    res = [acc[best_index],
           sen[best_index],
           spe[best_index],
           auc[best_index]]
    
    if len(y_pred) >= best_index + 1:
        return_y_pred = y_pred[best_index]
    else:
        return_y_pred = None
        
    return res, return_y_pred, best_index, len(acc)

def get_test_acc_using_val_best_itr(path, itr):
    acc, _, _, _ = read_file_metric_acc_sen_spe_f1(path)
    return acc[itr]
def get_val_metrics_and_test_accuracies(model, 
                                        val_fold_path, 
                                        ALL_BEST_ITR,
                                        ALL_TOTAL_ITERATION,
                                        ALL_Y_pred_in_test,
                                        based_best_metric='sensitivity', 
                                        subject_fold_name='LOO_', 
                                        SUBJECTALL=None, 
                                        total_subjects=64, 
                                        MAX_ITR=999,
                                        ):
    # get the averay validation result 
    ## only consider 1 iteration 
    num_of_cv_folds = len(os.listdir(val_fold_path + '/' + subject_fold_name + str(0)))#3 
    print('val_fold_path', val_fold_path + '/' + subject_fold_name + str(0))
    print('num_of_cv_folds', num_of_cv_folds)
    all_loo_metrics = []
    all_loo_acc = []
    LOOP_SUBJECT = SUBJECTALL if SUBJECTALL is not None else range(total_subjects)
    for loo in LOOP_SUBJECT: #range(total_subjects):
        ind_loo_folds =[]
        loo_acc = []
        # test_best_itr = []
        for cv_fold in range(num_of_cv_folds):
            # print(f"loo - {loo} - cv_fold - {cv_fold}")
            # read_fold = f"{val_fold_path}/LOO_{loo}/stratified_nested_{num_of_cv_folds}_CV_fold-{cv_fold}/"
            read_fold = f"{val_fold_path}/LOO_{loo}/stratified_nested_{num_of_cv_folds}_CV_fold-{cv_fold}/"
            read_val_path = read_fold + "val_acc.txt"
            read_test_path = read_fold + "test_acc.txt"

            res_metrics, y_pred, val_best_itr, total_itr = read_metrics_txt_best_itr(read_val_path, MAX_ITR, based_best_metric=based_best_metric)
            ALL_BEST_ITR.append(val_best_itr)
            ALL_TOTAL_ITERATION.append(total_itr)
            ALL_Y_pred_in_test.append(y_pred)
            ALL_inner_fold.append(res_metrics)
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
def check_if_all_subjects_are_trained(val_fold_path, TOTAL_Subject):
        return len(os.listdir(val_fold_path)) >= TOTAL_Subject


def modify_y_pred_by_giving_more_weight_to_1(ALL_Y_pred_in_test, K_FOLD, value_add_to_sensitivity=0.5):
    
    ALL_Y_pred_in_test = np.array(ALL_Y_pred_in_test)
    
    
    # --- using mean of the value 
    # ALL_Y_pred_in_test = np.reshape(ALL_Y_pred_in_test, (-1, K_FOLD, 2))
    # ALL_Y_pred_in_test = np.mean(ALL_Y_pred_in_test, axis=1)
    # y_pred_in_test_argmax = np.argmax(ALL_Y_pred_in_test, axis=1)
    # print('y_pred_in_test_argmax', y_pred_in_test_argmax)
    # --- 
    
    # --- previous -- slightly higher accuracy because of mean value will value some examples: true label = 1 but it is classified as 0.9999, 0.0001 in one fold in CV but other has a value like [0.4, 0.6] ... 
    
    ALL_Y_pred_in_test[:,1] += value_add_to_sensitivity
    y_pred_in_test_argmax = np.argmax(ALL_Y_pred_in_test, axis=1)
    y_pred_in_test_argmax = y_pred_in_test_argmax.reshape(-1, K_FOLD)
    y_pred_in_test_argmax = np.mean(y_pred_in_test_argmax, axis=1)
    
    # --- 
    y_pred_in_test_argmax = [1 if i >= 0.5 else 0 for i in y_pred_in_test_argmax]
    return y_pred_in_test_argmax

def avg_total_itr_for_each_fold(ALL_TOTAL_ITERATION, K_FOLD):
        loo_toal_itr = np.array(ALL_TOTAL_ITERATION).copy()
        loo_toal_itr = loo_toal_itr.reshape(-1, K_FOLD)
        loo_toal_itr = np.mean(loo_toal_itr, axis=1)
        return loo_toal_itr
def get_sorted_loo_array(model, model_params, TOTAL_Subject, DATASET, K_FOLD):

    ALL_TOTAL_ITERATION = [] # store all the total iteration for each fold
    # TOTAL_Subject = 64 # number of subjects in the dataset for LOOCV in external testing set
    validation_method_external = 'LOO_nested_CV' # external validation method
    validation_method_inner = f'stratified_nested_{K_FOLD}_CV_fold' # inner validation method
    # DATASET = 'prognosis/pre_treatment_hamd_reduction_50' # dataset name
    RESULT_FILE_NAME = 'val_acc.txt' # result file name
    val_fold_path = f'results/{model}/{DATASET}/{model_params}/{validation_method_external}'
    total_subjects  = 46 if DATASET[:4] == 'post' else TOTAL_Subject # '64' or '46


    for subject in range(total_subjects):
        for fold in range(K_FOLD):
            fold_path = f'{val_fold_path}/LOO_{subject}/{validation_method_inner}-{fold}'
            try:
                with open(f'{fold_path}/{RESULT_FILE_NAME}', 'r') as f:
                    total_lines = len(f.readlines())
                    ALL_TOTAL_ITERATION.append(total_lines)
            except:
                # if the fold has not been created yet, then the total iteration is 0
                print('fold_path is not founded yet', fold_path)
                # -100 means that in a loocv fold cv-fold 0~3 has been trained a lot of time but cv-fold 4 has not been trained yet, it will ask for training that fold first
                ALL_TOTAL_ITERATION.append(-100)
                print(f'{fold_path}/{RESULT_FILE_NAME} will be set to 0 because it has not been trained yet')
    # average the total iteration for each fold
    loo_toal_itr = avg_total_itr_for_each_fold(ALL_TOTAL_ITERATION, K_FOLD)
    sorted_indices = np.argsort(loo_toal_itr)
    sorted_indices = sorted_indices.tolist()

    print("Sorted indices:", sorted_indices, "Sorted values:", loo_toal_itr[sorted_indices])
    print(loo_toal_itr)
        
    # print("Sorted indices:", sorted_indices, "Sorted values:", loo_toal_itr[sorted_indices])
    return sorted_indices

def compute_save_MMDT_score(ALL_Y_pred_in_test, save_fold, num_of_k_fold=5):
    MMDT_score = ALL_Y_pred_in_test.copy()
    MMDT_score = np.array(MMDT_score)
    MMDT_score = np.argmax(MMDT_score, axis=-1)
    MMDT_score = MMDT_score.reshape(-1, num_of_k_fold)
    MMDT_score = np.mean(MMDT_score, axis=1)
    np.save(save_fold + '/MMDT_score.npy', MMDT_score)
    print('MMDT_score.shape', MMDT_score.shape)

def save_y_pred(y_pred, path, specify_itr, itr_version):
    if specify_itr is not None:
        path = path + f'/itr_{specify_itr}' 
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + f'/y_pred_{itr_version}.npy', y_pred)
    print("the y_pred has been saved in", path)

def get_average_of_y_pred(y_pred, K_FOLD=5):
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(-1, K_FOLD, 2)
    y_pred = np.mean(y_pred, axis=1)
    return y_pred
    

if __name__ == '__main__':
    TMP_ALL = []
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    # Add the arguments
    parser.add_argument('--max', type=int, required=True,
                        help='The maximum number of iterations')
    parser.add_argument('--result_path', type=str, required=True,
                        help='result_path')
    parser.add_argument('--K_FOLD', type=int, required=False,)
    parser.add_argument('--specify_itr', type=int, required=False,
                        help='specify the iteration, this value is conflict with the max value')
    parser.add_argument('--value_add_to_sensitivity_value', type=float, required=False,
                        default=0.0,
                        help='The value that will be added to the sensitivity value')
    # Parse the arguments
    args = parser.parse_args()
    K_FOLD = args.K_FOLD
    
    result_path = args.result_path
    result_path = result_path.split('/')
    model = result_path[1]
    data_prefix = result_path[2] + '/'
    dataset = result_path[3]
    save_path = '/'.join(result_path[:3])
    model_params = result_path[4]
    specify_itr = args.specify_itr
    MAX_ITR = args.max
    value_add_to_sensitivity_value = args.value_add_to_sensitivity_value
    total_subjects  = 46 if dataset[:4] == 'post' else 64 # '64' or '46

    itr_version = model_params[7:9]
    if itr_version[-1].isdigit():
        itr_version = int(itr_version)
    else:
        itr_version = int(itr_version[0])
        
    if not model_params:
        raise ValueError('Model name is not correct or there is no parameter for the model')
    SUBJECTALL = None #np.arange(4).tolist() + np.arange(30,34,1).tolist() + np.arange(49,55,1).tolist()# # np.arange(16).tolist()#None # np.arange(10).tolist() + np.arange(34,64).tolist()

    time = data_prefix + dataset
    # 'pre_treatment_hamd_reduction_50' or 'pre_post_treatment_hamd_reduction_50'

    validation_method = 'LOO_nested_CV'  # 'LOOCV' or 'k_fold' LOO_nested_CV
    based_best_metric = 'f1_score' # 'sensitivity' or 'f1_score'
    ALL_BEST_ITR = []
    ALL_TOTAL_ITERATION = []
    ALL_Y_pred_in_test = []
    ALL_inner_fold = [] # create this to see individual fold's result


    val_fold_path = f'results/{model}/{time}/{model_params}/LOO_nested_CV'
    
    output_fold = f'FigureTable/DL/timedomain/{time}'
    if not os.path.exists(output_fold):
        os.makedirs(output_fold)

    # y_test_path = f'allData/prognosis/{time}'
    y_test_path = f'allData/{data_prefix}' + dataset


    val_nested_CV_metrics, test_accuracy = get_val_metrics_and_test_accuracies(model, val_fold_path, ALL_BEST_ITR, ALL_TOTAL_ITERATION, ALL_Y_pred_in_test, based_best_metric=based_best_metric, SUBJECTALL=SUBJECTALL, total_subjects=total_subjects, MAX_ITR=MAX_ITR)
    
    
    y_test = np.load(y_test_path + '/label.npy')
    if SUBJECTALL is not None: y_test = y_test[SUBJECTALL]
    if model != 'fusion_catboost':
        print('ALL_Y_pred_in_test -> ', np.array(ALL_Y_pred_in_test).shape)
        y_pred_save_path = y_test_path + '/MDDR'
        save_y_pred(ALL_Y_pred_in_test, y_pred_save_path, specify_itr, f'loocv_v{itr_version}')
        y_pred_in_test_argmax = modify_y_pred_by_giving_more_weight_to_1(ALL_Y_pred_in_test, K_FOLD, value_add_to_sensitivity=value_add_to_sensitivity_value)
        
        compute_save_MMDT_score(ALL_Y_pred_in_test, y_test_path, K_FOLD)
            
        y_pred = y_pred_in_test_argmax
        
    else: 
        y_pred = convert_result_to_y_pred(test_accuracy, y_test)
    predict_accuracy_flag = y_pred==y_test
    y_pred_prob = get_average_of_y_pred(ALL_Y_pred_in_test)
    test_metrics = get_metrics_auc(y_test, y_pred_prob)

    print(f"MAX_ITR: {MAX_ITR} ranging ( {np.min(ALL_TOTAL_ITERATION)} ~ {np.max(ALL_TOTAL_ITERATION)} )")
    print('Model name:', model)
    print('specify_itr', specify_itr)
    
    print('value_add_to_sensitivity_value', value_add_to_sensitivity_value)
    print_md_table_val_test(model, test_metrics, val_nested_CV_metrics, using_AUC=True)
    print()
    
    ALL_inner_fold = np.array(ALL_inner_fold)
    ALL_inner_fold = ALL_inner_fold.reshape(-1, K_FOLD, 4)
    ALL_Y_pred_in_test_np = np.array(ALL_Y_pred_in_test)
    ALL_Y_pred_in_test_np = ALL_Y_pred_in_test_np.reshape(-1, K_FOLD, 2)
    for k in range(K_FOLD):
        mean_all_inner = np.mean(ALL_inner_fold[:,k,:], axis=0)
        k_y_pred = ALL_Y_pred_in_test_np[:,k,:]
        auc1 = round(roc_auc_score(y_test, k_y_pred[:,1]), 2)
        k_y_pred = np.argmax(k_y_pred, axis=-1)
        auc2 = round(roc_auc_score(y_test, k_y_pred), 2)
        k_test_metrics = get_metrics(y_test, k_y_pred)
        print_md_table_val_test(f'test_k_fold_{k}_AUC1_{auc1}_AUC2_{auc2}', k_test_metrics, mean_all_inner, print_table_header=False)
        print()


    loo_toal_itr = np.array(ALL_TOTAL_ITERATION).copy()
    loo_toal_itr = loo_toal_itr.reshape(-1, K_FOLD)
    loo_toal_itr = np.mean(loo_toal_itr, axis=1)
    sorted_indices = np.argsort(loo_toal_itr)
    sorted_indices = sorted_indices.tolist()
    print("Sorted indices:", sorted_indices, "Sorted values:", loo_toal_itr[sorted_indices])
    print(loo_toal_itr)    
    save_itr_version_with_test_val_auc(itr_version, test_metrics, val_nested_CV_metrics, save_path + '/metrics.txt')

    