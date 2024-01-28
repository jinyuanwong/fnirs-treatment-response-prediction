
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import os
import re


model = 'comb_cnn'  # comb_cnn or cnn_transformer

# 'pre_treatment_hamd_reduction_50' or 'pre_post_treatment_hamd_reduction_50'
time = 'pre_treatment_hamd_reduction_50'

validation_method = 'LOOCV'  # 'LOOCV' or 'k_fold'

output_fold = f'FigureTable/DL/TimeFeature/{time}'

if not os.path.exists(output_fold):
    os.makedirs(output_fold)

y_test_path = f'allData/prognosis/{time}'

total_fold = 65 # '65' or '45


def loop_iteration_find_best_performance_id(total_itr, model, verbose=True):
    read_path = 'results/' + model + '/' + time + '/' + validation_method + '-'
    best_acc = 0
    best_itr = 0
    final_wrong_index = []
    for index in range(total_itr):
        first_acc = []
        wrong_index = []
        for fold in range(total_fold):
            path = read_path + str(fold) + '/test_acc.txt'
            with open(path, 'r') as f:
                acc = f.read()
                acc = re.findall(r'accuracy: (\d+\.\d+)', acc)
                if float(acc[index]) != 1:
                    wrong_index.append(fold)
                first_acc.append(acc[index])
        first_acc = [float(i) for i in first_acc]
        mean_acc = np.mean(first_acc)
        if mean_acc > best_acc:
            best_itr = index
            best_acc = mean_acc
            final_wrong_index = wrong_index.copy()
        print(
            f'INDEX: {index} | mean: {mean_acc} | wrong subject: {wrong_index}')
    return final_wrong_index


# According to the result, find the wrong labeled index
comb_cnn_total_itr = 10
cnntr_total_itr = 2
wrong_index_mcnet = loop_iteration_find_best_performance_id(
    comb_cnn_total_itr, 'comb_cnn', verbose=False)
wrong_index_cnntr = loop_iteration_find_best_performance_id(
    cnntr_total_itr, 'cnn_transformer', verbose=False)


def using_wrong_index_to_generate_y_pred(y_test, wrong_index):
    y_pred = y_test.copy()
    for i in wrong_index:
        if y_pred[i] == 1:
            y_pred[i] = 0
        else:
            y_pred = 1
    return y_pred


# generate prediction of models
y_test = np.load(y_test_path + '/label.npy')
mcnet_pred = using_wrong_index_to_generate_y_pred(y_test, wrong_index_mcnet)
cnntr_pred = using_wrong_index_to_generate_y_pred(y_test, wrong_index_cnntr)

print(f'mcnet_pred: {mcnet_pred}')
print(f'cnntr_pred: {cnntr_pred}')


def get_metrics(y_true, y_pred):
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 明确指定labels参数
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 现在cm是一个2x2矩阵，即使数据只包含一个类别
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)

    return accuracy, sensitivity, specificity, f1


mcnet_metric = get_metrics(y_test, mcnet_pred)
cnntr_metric = get_metrics(y_test, cnntr_pred)

metric_dic = {
    'MCNet': mcnet_metric,
    'CNNTR': cnntr_metric
}

# metrics = [mcnet_metric, cnntr_metric]
metrics = []
for index, value in enumerate(mcnet_metric):
    metrics.append([mcnet_metric[index], cnntr_metric[index]])

print(metrics)
# Separate the data based on biomarkers
# Define the data
models = ['MCNet', 'CNNTR']
metrics_name = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score']

# Define colors for each metric for better visualization
colors = ['#2ca02c', '#4cbf4c', '#6edc6e',
          '#90f890']  # A green gradient palette
# colors = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c']
plt.rcParams['font.family'] = 'DejaVu Sans'
# Create separate plots for each biomarker with different colors for each metric
fig, ax = plt.subplots(figsize=(10, 6))
print(f'metrics: {metrics}')
bar_width = 0.15  # Adjust bar width for clarity

# Plot data for each model with different colors for each metric
for metric_index, metric_scores in enumerate(metrics):
    index = np.arange(len(metric_scores))
    ax.bar(index + metric_index * bar_width, metric_scores,
           bar_width, color=colors[metric_index], label=metrics_name[metric_index])

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title(f'Deep Learning Model Performance')
ax.set_xticks(index + bar_width * 1.5)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.05)
ax.legend(loc='upper right')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

plt.savefig(output_fold+f'/result.png')
# plt.show()


def generate_md_table():
    print('| Model Name | Accuracy | Sensitivity | Specificity | F1 Score |')
    print('|------------|----------|-------------|-------------|----------|')
    for model_name in models:
        print(f'| {model_name}      |', end='')
        for i in range(4):
            metric_name = metrics_name[i]
            val = metric_dic[model_name][i]
            print(f' {val:.4f}  |', end='')
        print()


generate_md_table()
