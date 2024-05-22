import numpy as np 
import os 
import sys 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix
from fine_tune_model import get_class_weight_dict
from fine_tune_model import tune_svm, tune_knn, tune_gradient_boosting, tune_adaboost, tune_xgboost, tune_mlp, tune_gaussian_nb
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import VotingClassifier
from validation_method import stratified_5_fold_classification, nested_cross_validation_classification, loocv_classification


def set_path():
    if sys.platform == 'darwin':
        print("Current system is macOS")
        main_fold_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction'
    elif sys.platform == 'linux':
        print("Current system is Ubuntu")
        main_fold_path = '/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning'
    else:
        print("Current system is neither macOS nor Ubuntu")
    os.chdir(main_fold_path)
    
def load_data():
    data_pth = 'allData/prognosis_mix_hb/pretreatment_response/nor_demo_cli_merge_data.npy'
    label_pth = 'allData/prognosis_mix_hb/pretreatment_response/label.npy'
    data = np.load(data_pth)
    labels = np.load(label_pth)
    return data, labels

def add_cgi(data):
    cgi_path = 'allData/prognosis_mix_hb/pretreatment_response/nor_T2_SDS_CGI.npy'
    cgi = np.load(cgi_path)
    data = np.concatenate((data, cgi[:, :6]), axis=1)
    return data

def add_mddr(data):
    mddr_path = 'allData/prognosis_mix_hb/pretreatment_response/MDDR/MDDR_derived_from_load_evaluate.npy'
    mddr = np.load(mddr_path)
    mddr = mddr[..., -1]
    mddr = np.transpose(mddr, (1, 0))
    data = np.concatenate((data, mddr), axis=1)
    return data
    
def get_class_weight_dict(labels):

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weight_dict= {i : class_weights[i] for i in range(len(class_weights))}
    print('class_weight_dict', class_weight_dict)
    # class_weight_dict[1] *= 10
    return class_weight_dict



def classification(data, labels):
    # Define the classifiers
    

    # classifiers = {
    #     "SVM": SVC(class_weight=get_class_weight_dict(labels), kernel='rbf', C=10, gamma='auto'), #tune_svm(data, labels), # 
    #     "XGBoost": XGBClassifier(scale_pos_weight=1e6), # tune_xgboost(data, labels), #
    #     "Naive Bayes": tune_gaussian_nb(data, labels),
    #     }
    
    svm_clf = tune_svm(data, labels)#SVC(class_weight=get_class_weight_dict(labels), kernel='rbf', C=10, gamma='auto', probability=True)
    xgb_clf = XGBClassifier(scale_pos_weight=1e6)
    nb_clf = tune_gaussian_nb(data, labels)
    
    # Define the Voting Classifier
    voting_clf = VotingClassifier(estimators=[
        ('SVM', svm_clf),
        ('XGBoost', xgb_clf),
        ('Naive Bayes', nb_clf)
    ], voting='hard')  # 'hard' for majority voting, 'soft' for weighted average probabilities
    
    # Define the classifiers dictionary including the Voting Classifier
    classifiers = {
        "Voting Classifier": voting_clf,
        "SVM": svm_clf,
        "XGBoost": xgb_clf,
        "Naive Bayes": nb_clf
    }
    loocv_classification(data, labels, classifiers)
    # stratified_5_fold_classification(data, labels, classifiers)
    # nested_cross_validation_classification(data, labels, classifiers)
    # Define the cross-validation str
if __name__ == "__main__":
    result_table = []
    # change the working directory to the main folder
    set_path()
    
    # load the data 
    data, labels = load_data()
    # data = add_cgi(data)
    data = add_mddr(data)

    print('data shape', data.shape)
    
    classification(data, labels)
    
    
    
    