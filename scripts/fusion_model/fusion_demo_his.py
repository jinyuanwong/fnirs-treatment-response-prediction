# first make sure the input data are good 
import numpy as np 
import os 
import sys
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, make_scorer, f1_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_validate, LeaveOneOut, StratifiedKFold
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import shap
if sys.platform == 'darwin':
    print("Current system is macOS")
    main_fold_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction'
elif sys.platform == 'linux':
    print("Current system is Ubuntu")
    main_fold_path = '/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning'
else:
    print("Current system is neither macOS nor Ubuntu")
    
sys.path.append(main_fold_path)    
os.chdir(main_fold_path)
from utils.hyperopt_utils import get_best_hyperparameters, get_best_hyperparameters_skf_inside_loocv_monitoring_recall_bacc
from utils.fnirs_utils import print_md_table_val_test_AUC

from scripts.fusion_model.fusion_model_utils import derive_average_MMDR_score
from scripts.fusion_model.fusion_model_utils import replace_nan_with_mean
from scripts.fusion_model.fusion_model_utils import impute_nan_data
from scripts.fusion_model.fusion_model_utils import process_with_nan_using_imputation_zscore
from scripts.fusion_model.fusion_model_utils import read_base_T2_SDS_CGI 
from scripts.fusion_model.fusion_model_utils import read_pychiatry
from scripts.fusion_model.fusion_model_utils import read_HAMD_score
from scripts.fusion_model.fusion_model_utils import read_demographic
from scripts.fusion_model.fusion_model_utils import plot_avg_auc
from scripts.fusion_model.fusion_model_utils import train_xgboost_shuffle_feature 
from scripts.fusion_model.fusion_model_utils import save_shap
import time
start_time = time.time()

K_FOLD = 5
fold_path = 'allData/prognosis_mix_hb/pretreatment_response'
MMDR_path = 'allData/prognosis_mix_hb/pretreatment_response/MDDR'

base_T2_SDS_CGI = read_base_T2_SDS_CGI(fold_path)
pyschiatry = read_pychiatry(fold_path)
HAMD_score = read_HAMD_score(fold_path)
demographic = read_demographic(fold_path)

pro_base_T2_SDS_CGI = process_with_nan_using_imputation_zscore(base_T2_SDS_CGI)
pro_pyschiatry = process_with_nan_using_imputation_zscore(pyschiatry)
pro_pyschiatry = np.concatenate((pro_pyschiatry[:, :-3], pro_pyschiatry[:, -2:]), axis=1) # must remove the -3rd column, because its existen will cause nan value of that column
pro_HAMD_score = process_with_nan_using_imputation_zscore(HAMD_score)
pro_demographic = process_with_nan_using_imputation_zscore(demographic)

fnirs_feature = derive_average_MMDR_score(MMDR_path, K_FOLD=K_FOLD)
Y = np.load(fold_path + '/label.npy', allow_pickle=True)

# repeat to see if seed is working 
data_name = 'his_metrics'
X_data = np.concatenate((pro_pyschiatry[:,:9], pro_demographic), axis=1)

shuffle_all_shaps = train_xgboost_shuffle_feature(X_data, 
                                                  Y, 
                                                  model_name='XGBoost',
                                                  num_shuffle=10, 
                                                  random_seed=1024,
                                                  title=f"Treatment Response Prediction (fNIRS + psychiatric history feature) ", 
                                                  is_plotting_avg_auc=True, 
                                                  is_shuffling=True, 
                                                  is_computing_shap=True,
                                                  best_params_xgboost=True,
                                                  num_evals=150,
                                                  loocv_metrics_save_file_name= data_name + '.npy')

save_shap(shuffle_all_shaps, X_data, output_fold='results/SHAP', name='shap_values_'+data_name+'.npy')
# End timing
end_time = time.time()

# Calculate total time taken
total_time = end_time - start_time
print(f"The program took {total_time} seconds to run.")
# nohup python scripts/fusion_model/fusion_demo_his.py > results/demo_his.log 2>&1 &