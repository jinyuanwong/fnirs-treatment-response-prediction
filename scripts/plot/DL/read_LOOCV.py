
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import os
import re 


model = 'comb_cnn'  # comb_cnn or cnn_transformer

# 'pre_treatment_hamd_reduction_50' or 'pre_post_treatment_hamd_reduction_50'
time = 'pre_treatment_hamd_reduction_50'

validation_method = 'LOOCV'  # 'LOOCV' or 'k_fold'

read_path = 'results/' + model + '/' + time + '/' + validation_method + '-'

total_fold = 65



for index in range(10):
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

    first_acc =  [float(i) for i in first_acc]
    mean_acc = np.mean(first_acc)
    print(f'INDEX: {index} | mean: {mean_acc} | wrong subject: {wrong_index}')