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

def classification(data, labels):
    # Define the classifiers
    classifiers = {
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "XGBoost": XGBClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    # Define the cross-validation strategy
    cv = StratifiedKFold(n_splits=5)

    # Initialize a dictionary to store the performance metrics
    metrics = ['roc_auc', 'accuracy', 'sensitivity', 'specificity']
    model_performance = {name: {metric: [] for metric in metrics} for name in classifiers}

    # Perform classification using each classifier
    for train_index, test_index in cv.split(data, labels):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_test, predictions)
            accuracy = accuracy_score(y_test, predictions)
            sensitivity = recall_score(y_test, predictions)
            tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
            specificity = tn / (tn + fp)
            
            # Store metrics
            model_performance[name]['roc_auc'].append(roc_auc)
            model_performance[name]['accuracy'].append(accuracy)
            model_performance[name]['sensitivity'].append(sensitivity)
            model_performance[name]['specificity'].append(specificity)
            
            print(f"{name} - ROC AUC: {roc_auc}, Accuracy: {accuracy}, Sensitivity: {sensitivity}, Specificity: {specificity}")
            
    # Calculate and print average performance metrics
    print("\nAverage performance:")
    for name, metrics in model_performance.items():
        averages = {metric: np.mean(scores) for metric, scores in metrics.items()}
        print(f"{name} - " + ", ".join(f"Average {metric.capitalize()}: {avg:.4f}" for metric, avg in averages.items()))

if __name__ == "__main__":
    
    # change the working directory to the main folder
    set_path()
    
    # load the data 
    # data, labels = load_data()
    
    # classification(data, labels)
    
    
    
    