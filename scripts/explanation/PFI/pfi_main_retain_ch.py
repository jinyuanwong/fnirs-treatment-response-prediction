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
from scripts.explanation.explanation_utils import show_ranked_shap_channel_importance
from LOO_nested_CV_model import build_model


model_name = 'gnn_transformer'
config_file_name = 'pretreatment_response_cv_5_mix_hb'
msg = 'loocv_v4to13'
TrainModel = build_model(model_name, config_file_name, msg)


from utils.utils_mine import stratified_LOO_nested_CV

data = TrainModel.data
label = TrainModel.label
adj = TrainModel.adj
current_loo = 0
k = 0
num_of_k_fold = 5
classifier = TrainModel.model
X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test = stratified_LOO_nested_CV(data, label, k=k, num_of_k_fold=num_of_k_fold, current_loo=current_loo, adj=adj)

print(f'X_train: {X_train.shape}')
print(f'X_val: {X_val.shape}')
print(f'X_test: {X_test.shape}')

def retain_importance(model, X_val_test, y, adj_test, num_iterations=1):
    # Store the baseline model performance
    baseline_performance = accuracy_score(y, model.predict([X_val_test, adj_test]).argmax(axis=1))
    
    # Initialize the importance dictionary
    importances = {}

    # Iterate over all features
    for i in range(X_val_test.shape[1]):
        feature_performance = []
        
        # Repeat the permutation process multiple times
        for _ in range(num_iterations):
            X_permuted = np.zeros_like(X_val_test)
            # Shuffle one feature (channel)
            X_permuted[:, i, :] = X_val_test[:, i, :]
            
            # Measure performance with the shuffled data
            permuted_performance = accuracy_score(y, model.predict([X_permuted, adj_test], verbose=False).argmax(axis=1))

            feature_performance.append(baseline_performance - permuted_performance)
        
        # Calculate the average drop in performance
        importances[i] = np.mean(feature_performance)
    
    return list(importances.values())


from sklearn.metrics import accuracy_score



def model_predict(model, X):
    return model.predict(X).argmax(axis=1)  # Modify according to your model's output

for v_itr in range(4,14):
    k_fold_all_importance = []
    for k_fold in range(0, 5):
        all_importance = []
        for subject in range(64):
            print('Current subject -> ', subject)
            X_train, Y_train, X_val, Y_val, X_test, Y_test, adj_train, adj_val, adj_test = stratified_LOO_nested_CV(data, label, k=k_fold, num_of_k_fold=num_of_k_fold, current_loo=subject, adj=adj)
            X_train = X_train[..., np.newaxis]
            
            X_val_test = np.concatenate([X_val, X_test], axis=0)[..., np.newaxis]
            Y_val_test = np.concatenate([Y_val, Y_test], axis=0).argmax(axis=1)
            adj_val_test = np.concatenate([adj_val, adj_test], axis=0)
            
            
            checkpoint_path = f'results/gnn_transformer/prognosis_mix_hb/pretreatment_response/loocv_v{v_itr}l1_rate_0.01_l2_rate_0.001_d_model_16_batch_size_64_n_layers_6/LOO_nested_CV/LOO_{subject}/stratified_nested_5_CV_fold-{k_fold}/fold-best-checkpoint'
            classifier.model.load_weights(checkpoint_path)
            model = classifier.model

            # importance = permutation_importance(model, X_val_test, Y_val_test, adj_val_test, num_iterations=2)
            importance = retain_importance(model, X_val_test, Y_val_test, adj_val_test, num_iterations=1)
            all_importance.append(importance)

        np_all_importance = np.array(all_importance)

        k_fold_all_importance.append(np_all_importance)
    np_k_fold_all_importance = np.array(k_fold_all_importance)

    np.save(f'results/gnn_transformer/prognosis_mix_hb/pretreatment_response/ten_itr_retain_single_ch_performance_v{v_itr}.npy', np_k_fold_all_importance)