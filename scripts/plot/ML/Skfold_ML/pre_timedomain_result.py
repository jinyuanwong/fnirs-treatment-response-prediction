# generate result from Skfold_CV_DMFC

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import os

timeline = 'pre'

# '{timeline}_treatment_hamd_reduction_50/{machine_learning_name}_result.npy'
input_fold = f'results/ML_results/timedomain/Skfold_CV_DMFC/'

output_fold = 'FigureTable/timedomain/Skfold_CV_DMFC/pre_treatment_hamd_reduction_50'

if not os.path.exists(output_fold):
    os.makedirs(output_fold)

def get_metrics(y_true, y_pred):
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 明确指定labels参数
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 现在cm是一个2x2矩阵，即使数据只包含一个类别
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0  # Or handle it in some other way, e.g., by setting it to 0 or to some default value

    f1 = f1_score(y_true, y_pred)

    return accuracy, sensitivity, specificity, f1

test_metrics = {}
validation_metrics = {}
ALL_HB_TYPE = ['HbO', 'HbR', 'HbO+HbR']
for machine_learning_name in ['Decision Tree', 'Random Forest', 'KNN', 'SVM']:
    
    data = np.load(
        f'{input_fold}/{timeline}_treatment_hamd_reduction_50/{machine_learning_name}_result.npy', allow_pickle=True).item()


    MODEL_HB_RES = {}
    VAL_MODEL_HB_RES = {}
    for HB_TYPE in ALL_HB_TYPE:
        sen = 0 
        val_sen = 0
        for time in range(5):
            time_hb = data[str(time)]['HB_TYPE_y_pred_and_y_test'][HB_TYPE][0]

            y_pred, y_test = time_hb[0], time_hb[1]
            
            # print('machine_learning_name:', machine_learning_name)
        
            metrics = get_metrics(y_test, y_pred)
            if metrics[1] >= sen:
                MODEL_HB_RES[HB_TYPE] = metrics
                sen = metrics[1]
                # print(f'current sensivity: {sen}')
            
            val_metrics = data[str(time)]['val_metrics_mean'][HB_TYPE]
            print(f'val_metrics-> {val_metrics}')
            if val_metrics[1] >= val_sen:
                VAL_MODEL_HB_RES[HB_TYPE] = val_metrics
                val_sen = val_metrics[1]
                # print(f'current val_sensivity: {val_sen}')

            
    test_metrics[machine_learning_name] = MODEL_HB_RES
    validation_metrics[machine_learning_name] = VAL_MODEL_HB_RES
    # print('metrics: ', metrics)

# Separate the data based on biomarkers
biomarkers = ['HbO', 'HbR', 'HbO+H']
# Define the data
models = ['Decision Tree', 'Random Forest', 'KNN', 'SVM']
metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score']

# Define colors for each metric for better visualization
colors = ['#2ca02c', '#4cbf4c', '#6edc6e',
          '#90f890']  # A green gradient palette
# colors = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c']
plt.rcParams['font.family'] = 'DejaVu Sans'
# Create separate plots for each biomarker with different colors for each metric

val_test_names = ['val', 'test']
models_metrics = [validation_metrics, test_metrics]
print(f'validation_metrics->{validation_metrics}')
for mindex, val_test in enumerate(val_test_names):
    model_metrics = models_metrics[mindex]
    for biomarker in biomarkers:
        # Extract data for the current biomarker
        model_data = [model_metrics[model][biomarker] for model in models]

        # Create a figure for the current biomarker
        fig, ax = plt.subplots(figsize=(10, 6))
        index = np.arange(len(models))
        bar_width = 0.15  # Adjust bar width for clarity

        # Plot data for each model with different colors for each metric
        for metric_index, metric in enumerate(metrics):
            metric_scores = [model[metric_index] for model in model_data]
            ax.bar(index + metric_index * bar_width, metric_scores,
                bar_width, color=colors[metric_index], label=metric)

        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title(f'{val_test} Model Performance on {biomarker}')
        ax.set_xticks(index + bar_width * 1.5)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        plt.savefig(output_fold+f'/{val_test}_{biomarker}.png')
        plt.show()
