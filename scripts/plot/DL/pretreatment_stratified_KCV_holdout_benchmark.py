
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import os
import re



# This is based on each iteration and find the best 
def loop_iteration_find_best_performance_id(total_itr, model, file_name, verbose=True):
    if specify_msg:
        read_path = 'results/' + model + '/' + time + '/' + specify_msg + '/' + validation_method + '-'
    else:
        read_path = 'results/' + model + '/' + time + '/' + validation_method + '-'    
    best_acc = 0
    best_itr = 0
    final_wrong_index = []
    print('| Iteration | Accuracy | Wrong Label Index |')
    print('| ---- | ---- | ---- |')
    for index in range(1,total_itr+1):
        folds_acc, folds_sen, folds_spe, folds_f1 = [], [], [], []
        
        wrong_index = []
        for fold in range(TOTAL_FOLD):
            path = read_path + str(fold) + '/' + file_name
            with open(path, 'r') as f:
                content = f.read()
                acc = re.findall(r'accuracy: (\d+\.\d+)', content)
                sen = re.findall(r'sensitivity: (\d+\.\d+)', content)
                spe = re.findall(r'specificity: (\d+\.\d+)', content)
                f1 = re.findall(r'F1-score: (\d+\.\d+)', content)
                
                folds_acc.append(acc[-index])
                folds_sen.append(sen[-index])
                folds_spe.append(spe[-index])
                folds_f1.append(f1[-index])
                
        folds_acc = [float(i) for i in folds_acc]
        folds_sen = [float(i) for i in folds_sen]
        folds_spe = [float(i) for i in folds_spe]
        folds_f1 = [float(i) for i in folds_f1]
        mean_acc = np.mean(folds_acc)
        if mean_acc > best_acc:
            best_itr = index
            best_acc = mean_acc
            best_metric = [np.mean(folds_acc), np.mean(folds_sen), np.mean(folds_spe), np.mean(folds_f1)]
        print(
            f' {index} | {folds_acc} | ')
    return best_metric

# Next - this is based on all the itertion and find the best 

# def loop_all_iteration_find_best_performance_id(total_itr, model, file_name, verbose=True):
#     if specify_msg:
#         read_path = 'results/' + model + '/' + time + '/' + specify_msg + '/' + validation_method + '-'
#     else:
#         read_path = 'results/' + model + '/' + time + '/' + validation_method + '-'

#     folds_acc, folds_sen, folds_spe, folds_f1 = [], [], [], []
#     for fold in range(TOTAL_FOLD):
#         acc_itr, sen_itr, spe_itr, f1_itr = [],[],[],[]
#         for index in range(1,total_itr+1):
#             path = read_path + str(fold) + '/' + file_name
#             with open(path, 'r') as f:
#                 content = f.read()
#                 acc = re.findall(r'accuracy: (\d+\.\d+)', content)
#                 sen = re.findall(r'sensitivity: (\d+\.\d+)', content)
#                 spe = re.findall(r'specificity: (\d+\.\d+)', content)
#                 f1 = re.findall(r'F1-score: (\d+\.\d+)', content)

                
#                 acc_itr.append(acc[-index])
#                 sen_itr.append(sen[-index])
#                 spe_itr.append(spe[-index])
#                 f1_itr.append(f1[-index])
                
#         acc_itr = [float(i) for i in acc_itr]
#         sen_itr = [float(i) for i in sen_itr]
#         spe_itr = [float(i) for i in spe_itr]
#         f1_itr = [float(i) for i in f1_itr]
#         based_on_best_metric_location = f1_itr
#         best_index = np.argmax(based_on_best_metric_location)
#         folds_acc.append(acc_itr[best_index])
#         folds_sen.append(sen_itr[best_index])
#         folds_spe.append(spe_itr[best_index])
#         folds_f1.append(f1_itr[best_index])
        
#     return np.mean(folds_acc), np.mean(folds_sen), np.mean(folds_spe), np.mean(folds_f1)


def loop_all_iteration_find_best_performance_id(total_itr, model, file_name, verbose=True):
    if specify_msg:
        read_path = 'results/' + model + '/' + time + '/' + specify_msg + '/' + validation_method + '-'
    else:
        read_path = 'results/' + model + '/' + time + '/' + validation_method + '-'

    folds_acc, folds_sen, folds_spe, folds_f1 = [], [], [], []
    for fold in range(TOTAL_FOLD):
        
        path = read_path + str(fold) + '/' + file_name
        with open(path, 'r') as f:
            content = f.read()
            acc = re.findall(r'accuracy: (\d+\.\d+)', content)
            sen = re.findall(r'sensitivity: (\d+\.\d+)', content)
            spe = re.findall(r'specificity: (\d+\.\d+)', content)
            f1 = re.findall(r'F1-score: (\d+\.\d+)', content)
        acc_itr, sen_itr, spe_itr, f1_itr = [],[],[],[]
        for index in range(1,total_itr+1):                
            acc_itr.append(float(acc[-index]))
            sen_itr.append(float(sen[-index]))
            spe_itr.append(float(spe[-index]))
            f1_itr.append(float(f1[-index]))
                
        based_on_best_metric_location = f1_itr
        best_index = np.argmax(based_on_best_metric_location)
        folds_acc.append(acc_itr[best_index])
        folds_sen.append(sen_itr[best_index])
        folds_spe.append(spe_itr[best_index])
        folds_f1.append(f1_itr[best_index])
        
    return np.mean(folds_acc), np.mean(folds_sen), np.mean(folds_spe), np.mean(folds_f1)



def refer_val_get_test(total_itr, model):
    if specify_msg:
        read_path = 'results/' + model + '/' + time + '/' + specify_msg + '/' + validation_method + '-'
    else:
        read_path = 'results/' + model + '/' + time + '/' + validation_method + '-'

    val_folds_acc, val_folds_sen, val_folds_spe, val_folds_f1 = [], [], [], []
    test_folds_acc, test_folds_sen, test_folds_spe, test_folds_f1 = [], [], [], []

    for fold in range(TOTAL_FOLD):
        
        path = read_path + str(fold) + '/' + val_file_name
        with open(path, 'r') as f:
            content = f.read()
            acc = re.findall(r'accuracy: (\d+\.\d+)', content)
            sen = re.findall(r'sensitivity: (\d+\.\d+)', content)
            spe = re.findall(r'specificity: (\d+\.\d+)', content)
            f1 = re.findall(r'F1-score: (\d+\.\d+)', content)
        val_acc_itr, val_sen_itr, val_spe_itr, val_f1_itr = [],[],[],[]
        for index in range(1,total_itr+1):                
            val_acc_itr.append(float(acc[-index]))
            val_sen_itr.append(float(sen[-index]))
            val_spe_itr.append(float(spe[-index]))
            val_f1_itr.append(float(f1[-index]))
        
        based_on_best_metric_location = val_sen_itr # val_f1_itr
        # sum_metrics = val_acc_itr + val_sen_itr + val_spe_itr + val_f1_itr
        # sum_metrics = [sum(values) for values in zip(val_acc_itr, val_sen_itr, val_spe_itr, val_f1_itr)]
        # based_on_best_metric_location = sum_metrics
        best_index = np.argmax(based_on_best_metric_location)
        path = read_path + str(fold) + '/' + test_file_name
        with open(path, 'r') as f:
            content = f.read()
            acc = re.findall(r'accuracy: (\d+\.\d+)', content)
            sen = re.findall(r'sensitivity: (\d+\.\d+)', content)
            spe = re.findall(r'specificity: (\d+\.\d+)', content)
            f1 = re.findall(r'F1-score: (\d+\.\d+)', content)
        test_acc_itr, test_sen_itr, test_spe_itr, test_f1_itr = [],[],[],[]
        for index in range(1,total_itr+1):                
            test_acc_itr.append(float(acc[-index]))
            test_sen_itr.append(float(sen[-index]))
            test_spe_itr.append(float(spe[-index]))
            test_f1_itr.append(float(f1[-index]))
        
        
        val_folds_acc.append(val_acc_itr[best_index])
        val_folds_sen.append(val_sen_itr[best_index])
        val_folds_spe.append(val_spe_itr[best_index])
        val_folds_f1.append(val_f1_itr[best_index])
        
        test_folds_acc.append(test_acc_itr[best_index])
        test_folds_sen.append(test_sen_itr[best_index])
        test_folds_spe.append(test_spe_itr[best_index])
        test_folds_f1.append(test_f1_itr[best_index])
        
    val_metrics = np.mean(val_folds_acc), np.mean(val_folds_sen), np.mean(val_folds_spe), np.mean(val_folds_f1)
    test_metrics = np.mean(test_folds_acc), np.mean(test_folds_sen), np.mean(test_folds_spe), np.mean(test_folds_f1)

    return val_metrics, test_metrics




def generate_md_table():
    print('| Model Name | Testing Set |             |             |             | Validation Set |             |             |             |')
    print('|------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|')
    print('|            | Accuracy | Sensitivity | Specificity | F1 Score | Accuracy | Sensitivity | Specificity | F1 Score |')


    # print('| Dataset | Model Name | Accuracy | Sensitivity | Specificity | F1 Score |')
    # print('|------------|------------|----------|-------------|-------------|----------|')
    print(f'| {model}   |', end='')
    for model_name in models:
        for i in range(4):
            metric_name = metrics_name[i]
            val = metric_dic[model_name][i]
            print(f' {val:.4f}  |', end='')
    print()
def count_lines_in_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return len(lines)
    except Exception as e:
        return str(e)
    
all_benchmark_models = ['chao_cfnn', 'li_svm', 'yu_gnn', 'zhu_xgboost', 'wang_alex']
model = 'wang_alex'  # comb_cnn or cnn_transformer or pre_post_cnn_transformer or gnn_transformer

# 'pre_treatment_hamd_reduction_50' or 'pre_post_treatment_hamd_reduction_50'
# DMFC/pre_treatment_hamd_reduction_50
time = 'prognosis/pretreatment_benchmarks'
# model = 'graphformer' # 'gnn_transformer'

condition_time = 'pre_post_treatment_hamd_reduction_50'

val_file_name = 'val_acc.txt'
test_file_name = 'test_acc.txt'


validation_method = 'Stratified_10_fold_CV/fold'  # 'LOOCV' or 'k_fold'
TOTAL_FOLD = 10

# According to the result, find the wrong labeled index
TOTAL_ITR = 10

# output_fold = f'FigureTable/DL/timedomain/{time}'

# if not os.path.exists(output_fold):
#     os.makedirs(output_fold)

folder_path = f'results/{model}/prognosis/pretreatment_benchmarks'

all_filename = os.listdir(folder_path)
# all_filename = ['d_model_64_BatchSize_4_n_layers_4']

for filename in all_filename:
    TOTAL_ITR = count_lines_in_file(folder_path + '/' + filename + '/' + validation_method + '-' + str(TOTAL_FOLD-1) + '/test_acc.txt')
    print(f'TOTAL_ITR->{TOTAL_ITR}')
    if TOTAL_ITR < 1:
        print("Hello End!")
        continue
    specify_msg = filename # 'l1_testIsDivBy3' # 'use_testset_divide_3', 'use_testset_divide_5' or None or 'use_testset_divide_7'


    val_best_metric, test_best_metric = refer_val_get_test(TOTAL_ITR, model)


    metric_dic = {
        # 'MCNet': mcnet_metric,
        'test_best_metric': test_best_metric,
        'val_best_metric': val_best_metric
    }

    # metrics = [mcnet_metric, cnntr_metric]
    metrics = []
    # for index, value in enumerate(mcnet_metric):
    #     metrics.append([mcnet_metric[index], cnntr_metric[index]])
        
    metrics = test_best_metric
    # Separate the data based on biomarkers
    # Define the data
    models = ['test_best_metric', 'val_best_metric'] # ['MCNet', 'CNNTR']
    metrics_name = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score']


    # if test_best_metric[1] > 0.4:
    print(f'dataset: {time}')
    print(f'model: {model}')
    print(f'validation method: stratified k-fold cross-validation with holdout')
    print(f'number of iterations: {TOTAL_ITR}')
    print(f'specify_msg: {specify_msg}')
    print()
    generate_md_table()