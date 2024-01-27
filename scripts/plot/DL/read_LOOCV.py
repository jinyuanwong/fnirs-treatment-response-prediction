
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import os
import re 


model = 'comb_cnn'  # comb_cnn or cnn_transformer

# 'pre_treatment_hamd_reduction_50' or 'pre_post_treatment_hamd_reduction_50'
time = 'pre_treatment_hamd_reduction_50',

validation_method = 'LOOCV'  # 'LOOCV' or 'k_fold'

read_path = 'results' + '/' + model + '/' + time + '/' + validation_method + '-'

total_fold = 64
for fold in range(total_fold):
    path = read_path + str(fold) + '/test.txt'
    with open(path, 'r') as f:
        acc = f.read()
        acc = re.findall(r'accuracy: (\d+\.\d+)', acc)
        print(acc)