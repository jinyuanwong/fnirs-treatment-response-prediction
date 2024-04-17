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

def derive_average_MMDR_score(input_fold, K_FOLD=5):
    all_X = []
    for i in range(1, 11):
        data_path = input_fold + f'/y_pred_loocv_v{i+3}.npy'
        X = np.mean(np.load(data_path).reshape(-1, K_FOLD, 2), axis=1)[:, 1:]
        all_X.append(X)
    all_X = np.array(all_X)
    all_X = np.mean(all_X, axis=0)
    return all_X

def replace_nan_with_mean(data):
    input = data.copy()
    for i in range(input.shape[1]):
        col = input[:, i]
        col = np.array(col, dtype=float)
        mean_value = np.nanmean(col)
        col[np.isnan(col)] = mean_value
        input[:, i] = col
    return input
def impute_nan_data(data):
    input = data.copy()
    from sklearn.experimental import enable_iterative_imputer
    # Now you can import IterativeImputer
    from sklearn.impute import IterativeImputer


    # Create an instance of IterativeImputer
    imputer = IterativeImputer()

    # Fit the imputer on the clinical_data ndarray
    imputer.fit(input)

    # Transform the clinical_data ndarray by imputing the NaN values
    imputed_data = imputer.transform(input)
    return imputed_data

def process_with_nan_using_imputation_zscore(data):
    data[data == 'missing data'] = np.nan
    data.astype(float)
    data = impute_nan_data(data)
    data = zscore(data, axis=0)
    return data

# CGI in T1 only has 1 value which is severity, so we delete them
def read_base_T2_SDS_CGI(fold_path):
    all_clinical = np.load(fold_path + '/CLINICAL_HISTORY.npy', allow_pickle=True)
    all_clinical_chosen_index = np.arange(0, 6).tolist() + np.arange(8, 16).tolist()
    all_clinical_chosen_index = np.array(all_clinical_chosen_index) 
    return all_clinical[:, all_clinical_chosen_index]


def read_pychiatry(fold_path):
    pyschiatry = np.load(fold_path + '/PSYCHIATRY_HISTORY.npy', allow_pickle=True)
    all_pyschiatry_chosen_index = np.arange(4).tolist() + np.arange(6, 15).tolist()
    all_pyschiatry_chosen_index = np.array(all_pyschiatry_chosen_index)
    return pyschiatry[:, all_pyschiatry_chosen_index]



def read_HAMD_score(fold_path):
    HAMD_ALL = np.load(fold_path + '/HAMD_ALL_HISTORY.npy', allow_pickle=True)
    HAMD_base_T2 = HAMD_ALL[:, :2]
    HAMD_reduction = (HAMD_ALL[:, 1] - HAMD_ALL[:, 0]) / HAMD_ALL[:, 0]
    HAMD_T1_T2_reduction = np.concatenate((HAMD_base_T2, HAMD_reduction[..., np.newaxis]), axis=1)
    return HAMD_T1_T2_reduction

def read_demographic(fold_path):
    demographic_data = np.load(fold_path + '/demographic.npy', allow_pickle=True)
    return demographic_data


# clinical_data = np.load(fold_path + '/baseline_clinical_data.npy', allow_pickle=True)
# demographic_data = np.load(fold_path + '/demografic_data.npy', allow_pickle=True)



def plot_avg_auc(fprs, tprs, roc_aucs, title):
    mean_fpr = np.linspace(0, 1, 100)

    # Interpolate TPRs at these common FPR levels
    mean_tpr = np.zeros_like(mean_fpr)
    tpr_interpolated = []

    for i in range(len(fprs)):
        tpr_interp = np.interp(mean_fpr, fprs[i], tprs[i])
        tpr_interpolated.append(tpr_interp)
        
        
    # Calculate the mean TPR
    tpr_interpolated = np.array(tpr_interpolated)
    mean_tpr = tpr_interpolated.mean(axis=0)
    std_tpr = tpr_interpolated.std(axis=0)

    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    # Compute AUC
    mean_auc = np.mean(roc_aucs) # auc(mean_fpr, mean_tpr)
    std_auc = np.std(roc_aucs)
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, lw=2, color='#000080', label='Mean ROC (AUC %0.2f ± %0.2f)' % (mean_auc, std_auc))
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3, label='±1 SD')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(title)
    plt.legend(loc="lower right", frameon=False)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.show()
    
    
def train_xgboost_shuffle_feature(X, 
                                  Y, 
                                  model_name='XGBoost',
                                  num_shuffle=2, 
                                  random_seed=1024,
                                  msg="", 
                                  title="", 
                                  is_plotting_avg_auc=False, 
                                  is_shuffling=True,
                                  is_computing_shap=True,
                                  best_params_xgboost=None,
                                  num_evals=10,
                                  loocv_metrics_save_file_name='fNIRS_demo_his_metrics.npy'):
    scale = 1e6# for scale in [1e6]: # [1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:
    

    model_dict = {
        'XGBoost': XGBClassifier(scale_pos_weight=scale, eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(scale_pos_weight=5, verbose=0),
    }
    np.random.seed(random_seed)
    ten_shuffle_seed = np.random.randint(0, 10000, num_shuffle)
    
    plt.figure(figsize=(10, 10))

    def specificity_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        return specificity

    def get_f1_score(y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')
    
    # Create a scorer for specificity
    specificity_scorer = make_scorer(specificity_score)
    get_f1_scorer = make_scorer(get_f1_score)

    scoring = {'balanced_accuracy': make_scorer(balanced_accuracy_score),
                'sensitivity': make_scorer(recall_score),
                'specificity': specificity_scorer,
                'f1_score': get_f1_scorer,
                'AUC': make_scorer(roc_auc_score)}
    # Assuming X, Y, and a dictionary of models are already defined

    # Outer loop: Leave-One-Out Cross-Validation (LOOCV)
    loo = LeaveOneOut()

    fprs, tprs, roc_aucs = [], [], []
    shuffle_inner_fold = []
    shuffle_outer_fold = []
    if is_computing_shap: shuffle_all_shaps = []
    
    if not is_shuffling: num_shuffle = 1
    
    for shuffle_i in range(num_shuffle):
     
        np.random.seed(ten_shuffle_seed[shuffle_i])
        y_pred_all = []
        results = {}
        all_inner_fold = []
        all_outer_fold = []
        all_shaps = []
        y_pred_prob_all = []

        # Shuffle X_tmp and Y_tmp
        if is_shuffling:
            shuffled_indices = np.random.permutation(X.shape[1])
        else:
            shuffled_indices = np.arange(X.shape[1])
        print(" shuffled_indices ", shuffled_indices)
        
        X_tmp_shuffled = X[:,shuffled_indices]
        Y_tmp_shuffled = Y
        
        if best_params_xgboost is not None:
            best_params_xgboost = get_best_hyperparameters_skf_inside_loocv_monitoring_recall_bacc(X_tmp_shuffled, Y_tmp_shuffled, num_evals=num_evals, random_seed=ten_shuffle_seed[shuffle_i])
            model_dict['XGBoost'] = XGBClassifier(**best_params_xgboost)               

        # for model_name, model in models.items():
        # model_name = 'XGBoost'
        model = model_dict[model_name]
        

        for train_index, test_index in loo.split(X):
            # Splitting the dataset for this LOOCV iteration
            X_train, X_test = X_tmp_shuffled[train_index], X_tmp_shuffled[test_index]
            Y_train, Y_test = Y_tmp_shuffled[train_index], Y_tmp_shuffled[test_index]
            stratified_kfold = StratifiedKFold(n_splits=5)
            cv_results = cross_validate(model, X_train, Y_train, cv=stratified_kfold, scoring=scoring, return_train_score=False)
        
            # Calculate mean scores
            mean_accuracy = cv_results['test_balanced_accuracy'].mean()
            mean_sensitivity = cv_results['test_sensitivity'].mean()
            mean_specificity = cv_results['test_specificity'].mean()
            mean_AUC = cv_results['test_AUC'].mean()
            # m
            
            # Store the results
            results[model_name] = {'bAcc': mean_accuracy, 'Sensitivity': mean_sensitivity, 'Specificity': mean_specificity, 'AUC': mean_AUC}
            all_inner_fold.append([results[model_name]['bAcc'], results[model_name]['Sensitivity'], results[model_name]['Specificity'], results[model_name]['AUC']])
            # print('inner cv results', results[model_name])
            # Inner loop: 5-fold cross-validation for training and validating the model
            # Note: Adjustments might be needed depending on how you intend to use the results of inner CV for model selection or tuning
            
            # Train the model using the inner CV folds
            # Note: You might use cross_val_predict or another approach here for model selection or tuning
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)

            probabilities = model.predict_proba(X_test)
            y_pred_prob_all.append(probabilities)
            y_pred_all.append(Y_pred)
            
            if is_computing_shap:
                # Create an explainer object
                explainer = shap.TreeExplainer(model)
                # Calculate SHAP values for all samples
                shap_values = explainer(X_test) 
                all_shaps.append(shap_values)

            # Assuming X, Y, and a trained XGBoost model are already defined        

        y_pred_ind = []
        for i in range(len(y_pred_all)):
            y_pred_ind.append(y_pred_all[i][0])


        
        f1 = f1_score(Y_tmp_shuffled, y_pred_ind, average='weighted')
        accuracy = accuracy_score(Y_tmp_shuffled, y_pred_ind)
        # Calculate sensitivity (recall)
        sensitivity = recall_score(Y_tmp_shuffled, y_pred_ind)  # This is already done in your snippet
        # Calculate specificity
        specificity = specificity_score(Y_tmp_shuffled, y_pred_ind)
        np_y_pred_prob_all = np.array([i[0] for i in y_pred_prob_all])

        fpr, tpr, _ = roc_curve(Y_tmp_shuffled, np_y_pred_prob_all[:, 1])
        roc_auc = auc(fpr, tpr)
        
        all_outer_fold = [accuracy, sensitivity, specificity, roc_auc]
        shuffle_outer_fold.append(all_outer_fold)
        print(f"num_of_shuffle {shuffle_i} scaled by {scale} Accuracy: {accuracy} Sensitivity is {sensitivity} Specificity is {specificity} AUC is {roc_auc}")

        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f) } - random - %d' % (roc_auc, shuffle_i) + msg)

        all_inner_fold = np.array(all_inner_fold)
        all_inner_fold_mean = np.mean(all_inner_fold, axis=0)
        shuffle_inner_fold.append(all_inner_fold_mean)
        print(all_inner_fold_mean)
        

        if is_computing_shap: shuffle_all_shaps.append(all_shaps)
        
    # Plotting the ROC curve

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(title)
    plt.legend(loc="lower right")
    
    if is_plotting_avg_auc:
        plot_avg_auc(fprs, tprs, roc_aucs, title)
        loocv_metrics = {'fprs': fprs, 'tprs': tprs, 'roc_aucs': roc_aucs}
        np.save('results/' + loocv_metrics_save_file_name, loocv_metrics)
    # plt.show()
    
    mean_shuffle_inner_fold = np.mean(shuffle_inner_fold, axis=0)
    mean_shuffle_outer_fold = np.mean(shuffle_outer_fold, axis=0)

    std_shuffle_inner_fold = np.std(shuffle_inner_fold, axis=0)
    std_shuffle_outer_fold = np.std(shuffle_outer_fold, axis=0)
    
    print_md_table_val_test_AUC('Mean ' + model_name, mean_shuffle_outer_fold, mean_shuffle_inner_fold)
    print_md_table_val_test_AUC('SD ' + model_name, std_shuffle_outer_fold, std_shuffle_inner_fold)

    if is_computing_shap: return shuffle_all_shaps
    else: return None


def save_shap(shuffle_all_shaps, X_data, output_fold='results/SHAP', name='shap_values_fnirs_demographic_pyschiatry.npy'):
    shap_values = []
    shap_base_values = []
    num_subject = X_data.shape[0]
    num_features = X_data.shape[1]
    for i in range(len(shuffle_all_shaps)):
        for j in range(num_subject):
                shap_v = []
                shap_b = []
                for k in range(num_features):  
                    shap_v.append(shuffle_all_shaps[i][j][0][k].values)
                    shap_b.append(shuffle_all_shaps[i][j][0][k].base_values)
                shap_values.append(shap_v)
                shap_base_values.append(shap_b)

    shap_values = np.array(shap_values) / np.max(np.abs(shap_values))
    shap_values = shap_values.reshape(-1, num_subject, num_features)
    if not os.path.exists(output_fold):
        os.makedirs(output_fold)
        
    np.save(output_fold + '/' + name, shap_values)


fold_path = 'allData/prognosis_mix_hb/pretreatment_response'
MMDR_path = 'allData/prognosis_mix_hb/pretreatment_response/MDDR'

base_T2_SDS_CGI = read_base_T2_SDS_CGI(fold_path)
pyschiatry = read_pychiatry(fold_path)
HAMD_score = read_HAMD_score(fold_path)
demographic = read_demographic(fold_path)

pro_base_T2_SDS_CGI = process_with_nan_using_imputation_zscore(base_T2_SDS_CGI)
pro_pyschiatry = process_with_nan_using_imputation_zscore(pyschiatry)
print('pro_pyschiatry ->', pro_pyschiatry.shape)
pro_pyschiatry = np.concatenate((pro_pyschiatry[:, :-3], pro_pyschiatry[:, -2:]), axis=1) # must remove the -3rd column, because its existen will cause nan value of that column
pro_HAMD_score = process_with_nan_using_imputation_zscore(HAMD_score)
pro_demographic = process_with_nan_using_imputation_zscore(demographic)

K_FOLD = 5
MMDR_path = 'allData/prognosis_mix_hb/pretreatment_response/MDDR'
fnirs_feature = derive_average_MMDR_score(MMDR_path, K_FOLD=K_FOLD)
fnirs_feature = derive_average_MMDR_score(MMDR_path) # np.mean(np.load(MMDR_path + '/y_pred_loocv_v4.npy').reshape(-1, K_FOLD, 2), axis=1)[:, 1:]

Y = np.load(fold_path + '/label.npy', allow_pickle=True)

# repeat to see if seed is working 
data_name = 'fNIRS_demo_metrics'
X_data = np.concatenate((pro_demographic, fnirs_feature), axis=1)

shuffle_all_shaps = train_xgboost_shuffle_feature(X_data, 
                                                  Y, 
                                                  model_name='XGBoost',
                                                  num_shuffle=10, 
                                                  random_seed=1024,
                                                  title=f"Treatment Response Prediction (fNIRS + demographic and psychiatric feature) ", 
                                                  is_plotting_avg_auc=True, 
                                                  is_shuffling=True, 
                                                  is_computing_shap=True,
                                                  best_params_xgboost=True,
                                                  num_evals=150,
                                                  loocv_metrics_save_file_name= data_name + '.npy')

save_shap(shuffle_all_shaps, X_data, output_fold='results/SHAP', name='shap_values_'+data_name+'.npy')


