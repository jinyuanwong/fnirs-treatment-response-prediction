
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import os
import re


model = 'gnn_transformer'  # comb_cnn or cnn_transformer or pre_post_cnn_transformer or gnn_transformer

# 'pre_treatment_hamd_reduction_50' or 'pre_post_treatment_hamd_reduction_50'
time = 'pre_treatment_hamd_reduction_50'

specify_msg = 'l1_testIsDivBy3' # 'use_testset_divide_5' or None or 'use_testset_divide_7'

condition_time = 'pre_post_treatment_hamd_reduction_50'

val_file_name = 'val_acc.txt'
test_file_name = 'test_acc.txt'

if time[-len(condition_time):] == 'pre_post_treatment_hamd_reduction_50':
    validation_method = 'Stratified_4_fold_CV/fold'  # 'LOOCV' or 'k_fold'
    total_fold = 4 # '65' or '46

else:
    validation_method = 'Stratified_5_fold_CV/fold'  # 'LOOCV' or 'k_fold'
    total_fold = 5

# According to the result, find the wrong labeled index
comb_cnn_total_itr = 10
cnntr_total_itr = 4

output_fold = f'FigureTable/DL/timedomain/{time}'

if not os.path.exists(output_fold):
    os.makedirs(output_fold)

y_test_path = f'allData/prognosis/{time}'

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
        for fold in range(total_fold):
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
#     for fold in range(total_fold):
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
    for fold in range(total_fold):
        
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

    for fold in range(total_fold):
        
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
        
        based_on_best_metric_location = val_f1_itr
        # sum_metrics = val_acc_itr + val_sen_itr + val_spe_itr + val_f1_itr
        # sum_metrics = [sum(values) for values in zip(val_acc_itr, val_sen_itr, val_spe_itr, val_f1_itr)]
        # based_on_best_metric_location = sum_metrics
        best_index = np.argmax(based_on_best_metric_location)
        print(f'best_index_from_sum:{best_index}')        
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

# wrong_index_mcnet = loop_iteration_find_best_performance_id(
#     comb_cnn_total_itr, 'comb_cnn', verbose=False)
# val_best_metric = loop_iteration_find_best_performance_id(
#     cnntr_total_itr, model, 'val_acc.txt',verbose=False)


# test_best_metric = loop_all_iteration_find_best_performance_id(
#     cnntr_total_itr, model, 'test_acc.txt',verbose=False)
# print(f'test_best-metric: {test_best_metric}')
# print('-'*100)
# val_best_metric = loop_all_iteration_find_best_performance_id(
#     cnntr_total_itr, model, 'val_acc.txt',verbose=False)

val_best_metric, test_best_metric = refer_val_get_test(cnntr_total_itr, model)


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

print(metrics)
# Separate the data based on biomarkers
# Define the data
models = ['test_best_metric', 'val_best_metric'] # ['MCNet', 'CNNTR']
metrics_name = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score']

# # Define colors for each metric for better visualization
# colors = ['#2ca02c', '#4cbf4c', '#6edc6e',
#           '#90f890']  # A green gradient palette
# # colors = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c']
# plt.rcParams['font.family'] = 'DejaVu Sans'
# # Create separate plots for each biomarker with different colors for each metric
# fig, ax = plt.subplots(figsize=(10, 6))
# print(f'metrics: {metrics}')
# bar_width = 0.15  # Adjust bar width for clarity

# # Plot data for each model with different colors for each metric
# for metric_index, metric_scores in enumerate(metrics):
#     index = np.arange(1)
#     ax.bar(index + metric_index * bar_width, metric_scores,
#            bar_width, color=colors[metric_index], label=metrics_name[metric_index])

# ax.set_xlabel('Model')
# ax.set_ylabel('Score')
# ax.set_title(f'Deep Learning Model Performance')
# ax.set_xticks(index + bar_width * 1.5)
# ax.set_xticklabels(models)
# ax.set_ylim(0, 1.05)
# ax.legend(loc='upper right')

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# plt.tight_layout()

# plt.savefig(output_fold+f'/result.png')
# plt.show()

print(f'dataset: {time}')
print(f'model: {model}')
print(f'validation method: stratified k-fold cross-validation with holdout')
print(f'number of iterations: {cnntr_total_itr}')
print(f'specify_msg: {specify_msg}')
print()

def generate_md_table():
    print('| Model Name | Accuracy | Sensitivity | Specificity | F1 Score |')
    print('|------------|----------|-------------|-------------|----------|')
    
    # print('| Dataset | Model Name | Accuracy | Sensitivity | Specificity | F1 Score |')
    # print('|------------|------------|----------|-------------|-------------|----------|')
    
    for model_name in models:
        print(f'| {model_name}      |', end='')
        for i in range(4):
            metric_name = metrics_name[i]
            val = metric_dic[model_name][i]
            print(f' {val:.4f}  |', end='')
        print()

generate_md_table()
