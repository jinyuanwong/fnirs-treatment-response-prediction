
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import os
import re


model = 'pre_post_cnn_transformer'  # comb_cnn or cnn_transformer or pre_post_cnn_transformer or gnn_transformer

# 'pre_treatment_hamd_reduction_50' or 'pre_post_treatment_hamd_reduction_50'
time = 'pre_post_treatment_hamd_reduction_50'

condition_time = 'pre_post_treatment_hamd_reduction_50'

if time[-len(condition_time):] == 'pre_post_treatment_hamd_reduction_50':
    validation_method = 'Stratified_4_fold_CV/fold'  # 'LOOCV' or 'k_fold'
    total_fold = 4 # '65' or '45

else:
    validation_method = 'Stratified_5_fold_CV/fold'  # 'LOOCV' or 'k_fold'
    total_fold = 5

# According to the result, find the wrong labeled index
comb_cnn_total_itr = 10
cnntr_total_itr = 10

output_fold = f'FigureTable/DL/TimeFeature/{time}'

if not os.path.exists(output_fold):
    os.makedirs(output_fold)

y_test_path = f'allData/prognosis/{time}'

def loop_iteration_find_best_performance_id(total_itr, model, file_name, verbose=True):
    read_path = 'results/' + model + '/' + time + '/' + validation_method + '-'
    best_acc = 0
    best_itr = 0
    final_wrong_index = []
    print('| Iteration | Accuracy | Wrong Label Index |')
    print('| ---- | ---- | ---- |')
    for index in range(total_itr):
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
                
                folds_acc.append(acc[index])
                folds_sen.append(sen[index])
                folds_spe.append(spe[index])
                folds_f1.append(f1[index])
                
        folds_acc = [float(i) for i in folds_acc]
        folds_sen = [float(i) for i in folds_sen]
        folds_spe = [float(i) for i in folds_spe]
        folds_f1 = [float(i) for i in folds_f1]
        mean_acc = np.mean(folds_f1)
        if mean_acc > best_acc:
            best_itr = index
            best_acc = mean_acc
            best_metric = [np.mean(folds_acc), np.mean(folds_sen), np.mean(folds_spe), np.mean(folds_f1)]
        print(
            f' {index} | {folds_acc} | ')
    return best_metric



# wrong_index_mcnet = loop_iteration_find_best_performance_id(
#     comb_cnn_total_itr, 'comb_cnn', verbose=False)
test_best_metric = loop_iteration_find_best_performance_id(
    cnntr_total_itr, model, 'test_acc.txt',verbose=False)
print('-'*100)
val_best_metric = loop_iteration_find_best_performance_id(
    cnntr_total_itr, model, 'val_acc.txt',verbose=False)



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
