
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
import os

from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, recall_score

import xgboost as xgb
"""
this file name: Skfold_CV_DMFC.py

so this file will generate metrics for each model, and save them to the output_fold

"""

output_fold = 'results/ML_results/timedomain/LOO_nested_CV_timedomain/pre_treatment_hamd_reduction_50'

input_path = 'allData/prognosis/pre_treatment_hamd_reduction_50'

data = np.load(input_path + '/data.npy')
label = np.load(input_path + '/label.npy')


if not os.path.exists(output_fold):
    os.makedirs(output_fold)


res = {}

# 初始化模型，同时设置随机种子
models = {
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    # "Logistic Regression": LogisticRegression(max_iter=150),
    
    # "Random Forest": RandomForestClassifier(),
    # "SVM": SVC(),
    # "KNN": KNeighborsClassifier(),  # 注意：KNN通常没有random_state参数
    # "Decision Tree": DecisionTreeClassifier()
}
for name, model in models.items():
    # run multiple time, using different time stamp as random seed
    for num_time in range(5):
        print(f'current training {name}, num_time: {num_time}')

        # 使用当前时间戳作为随机种子
        current_time_seed = int(time.time()) + num_time

        # build model
        model.random_state = current_time_seed

        hb_result = {}
        HB_TYPE_y_pred_and_y_test = {}
        VAL_metrics_mean = {}
        VAL_metrics_std = {}
        for HB_TYPE in ['HbO', 'HbR', 'HbO+HbR']:
            HB_TYPE_y_pred_and_y_test[HB_TYPE] = []
            VAL_metrics_mean[HB_TYPE] = []
            VAL_metrics_std[HB_TYPE] = []

            if HB_TYPE == 'HbO':
                hb = data[..., :data.shape[-1]//2]
            elif HB_TYPE == 'HbR':
                hb = data[..., data.shape[-1]//2:]
            else:
                hb = data.copy()
            hb_2d = np.reshape(hb, (hb.shape[0], -1))
            
            loo = LeaveOneOut()

            # Apply LOOCV to train the model
            # Initialize LeaveOneOut

            # 存储每个模型的准确率
            accuracies = {}
            # Loop over each train/test split
            for train_index, test_index in loo.split(hb_2d):
                # Split the data into training and testing sets
                X_train, X_test = hb_2d[train_index], hb_2d[test_index]
                y_train, y_test = label[train_index], label[test_index]
                num_of_k_fold = int((label==1).sum() * 2 / 3 / 2)
            
            
                # ----------------This is for accuracy + recall matrics----------------------
                # scoring = {
                #     'accuracy': make_scorer(accuracy_score),
                #     'recall': make_scorer(recall_score),
                #     'f1_score': make_scorer(f1_score),
                #     'specificity': make_scorer(lambda y_true, y_pred: confusion_matrix(y_true, y_pred).ravel()[0] / 
                #                             (confusion_matrix(y_true, y_pred).ravel()[0] + confusion_matrix(y_true, y_pred).ravel()[1]))
                # }
                # # Perform cross-validation
                # cv_results = cross_validate(model, X_train, y_train, cv=num_of_k_fold, scoring=scoring)

                # # Print the accuracy and recall for each fold
                # print("Accuracy for each fold: ", cv_results['test_accuracy'])
                # print("Recall (Sensitivity) for each fold: ", cv_results['test_recall'])

                # # Calculate mean and standard deviation for accuracy and recall
                # mean_accuracy = cv_results['test_accuracy'].mean()
                # std_accuracy = cv_results['test_accuracy'].std() * 2  # 95% confidence interval

                # mean_recall = cv_results['test_recall'].mean()
                # std_recall = cv_results['test_recall'].std() * 2  # 95% confidence interval
                
                # mean_specificity = cv_results['test_specificity'].mean()
                # std_specificity = cv_results['test_specificity'].std() * 2  # 95% confidence interval
                
                # mean_f1_score = cv_results['test_f1_score'].mean()
                # std_f1_score = cv_results['test_f1_score'].std() * 2  # 95% confidence interval            
                # #--------------------------------------
                # # Print the mean accuracy and recall with 95% confidence intervals
                # print("Mean accuracy: %0.2f (± %0.2f)" % (mean_accuracy, std_accuracy))
                # print("Mean recall (Sensitivity): %0.2f (± %0.2f)" % (mean_recall, std_recall))
                #----------------This is for simple accuracy matrics----------------------
                
                #--------------------------------------
                model.fit(X_train, y_train)
                # Predict the labels for the test set
                y_pred = model.predict(X_test)


                # VAL_metrics_mean[HB_TYPE].append([mean_accuracy, mean_recall, mean_specificity, mean_f1_score])
                # VAL_metrics_std[HB_TYPE].append([std_accuracy, std_recall, std_specificity, std_f1_score])
                HB_TYPE_y_pred_and_y_test[HB_TYPE].append([y_pred, y_test])
        
        save_result={}
        save_result['VAL_metrics_mean'] = VAL_metrics_mean
        save_result['VAL_metrics_std'] = VAL_metrics_std
        save_result['current_time_seed'] = current_time_seed
        save_result['num_time'] = num_time
        save_result['HB_TYPE_y_pred_and_y_test'] = HB_TYPE_y_pred_and_y_test

        res[f'{num_time}'] = save_result
        print(res)
    np.save(output_fold + f'/{name}_result.npy', res)
