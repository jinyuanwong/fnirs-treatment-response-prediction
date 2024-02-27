
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


"""
this file name: Skfold_CV_DMFC.py

so this file will generate metrics for each model, and save them to the output_fold

"""

output_fold = 'results/ML_results/timedomain/Skfold_CV_DMFC/pre_treatment_hamd_reduction_50'

input_path = 'allData/prognosis/pre_treatment_hamd_reduction_50'

data = np.load(input_path + '/data.npy')
label = np.load(input_path + '/label.npy')


def stratified_k_fold_cross_validation_with_holdout(data, label):
    label_not_onehot = label
    pos = data[label_not_onehot==1]
    neg = data[label_not_onehot==0]
    
    holdout_pos_num = pos.shape[0] // 3
    holdout_neg_num = neg.shape[0] // 3
    
    X_test = np.concatenate((pos[:holdout_pos_num], neg[:holdout_neg_num]), axis=0)
    Y_test = np.concatenate((np.ones(holdout_pos_num), np.zeros(holdout_neg_num)), axis=0)
    
    train_val_pos = pos[holdout_pos_num:]
    train_val_neg = neg[holdout_neg_num:]
    
    X_train = np.concatenate((train_val_pos, train_val_neg), axis=0)
    Y_train = np.concatenate((np.ones(train_val_pos.shape[0]), np.zeros(train_val_neg.shape[0])), axis=0)

    """
    train_val_pos_num 
    - should be 10 for pretreatment dataset 
    - should be 8 for pre-post-treatment dataset 
    
    next, I will devide them by 2 to do 5 and 4 fold cross validation
    so the num_of_k_fold should be calulated by 
    (label==1).sum() * 2 / 3 / 2
    """

    return X_train, Y_train, X_test, Y_test

    


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
    specificity = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)

    return accuracy, sensitivity, specificity, f1


res = {}

# 初始化模型，同时设置随机种子
models = {
    # "Logistic Regression": LogisticRegression(max_iter=150),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),  # 注意：KNN通常没有random_state参数
    "Decision Tree": DecisionTreeClassifier()
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
        HB_TYPE_accuraies = {}
        HB_TYPE_y_pred_and_y_test = {}
        val_metrics_mean = {}
        val_metrics_std = {}
        for HB_TYPE in ['HbO', 'HbR', 'HbO+HbR']:
            HB_TYPE_accuraies[HB_TYPE] = []
            HB_TYPE_y_pred_and_y_test[HB_TYPE] = []

            if HB_TYPE == 'HbO':
                hb = data[..., :data.shape[-1]//2]
            elif HB_TYPE == 'HbR':
                hb = data[..., data.shape[-1]//2:]
            else:
                hb = data.copy()
            hb_2d = np.reshape(hb, (hb.shape[0], -1))

            # Apply LOOCV to train the model
            # Initialize LeaveOneOut

            # 存储每个模型的准确率
            accuracies = {}

            num_of_k_fold = int((label==1).sum() * 2 / 3 / 2)
            X_train,y_train,X_test,y_test = stratified_k_fold_cross_validation_with_holdout(hb_2d,label)
            
            
            #----------------This is for accuracy + recall matrics----------------------
            scoring = {'accuracy': make_scorer(accuracy_score), 'recall': make_scorer(recall_score)}
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'recall': make_scorer(recall_score),
                'f1_score': make_scorer(f1_score),
                'specificity': make_scorer(lambda y_true, y_pred: confusion_matrix(y_true, y_pred).ravel()[0] / 
                                        (confusion_matrix(y_true, y_pred).ravel()[0] + confusion_matrix(y_true, y_pred).ravel()[1]))
            }
            # Perform cross-validation
            cv_results = cross_validate(model, X_train, y_train, cv=num_of_k_fold, scoring=scoring)

            # Print the accuracy and recall for each fold
            print("Accuracy for each fold: ", cv_results['test_accuracy'])
            print("Recall (Sensitivity) for each fold: ", cv_results['test_recall'])

            # Calculate mean and standard deviation for accuracy and recall
            mean_accuracy = cv_results['test_accuracy'].mean()
            std_accuracy = cv_results['test_accuracy'].std() * 2  # 95% confidence interval

            mean_recall = cv_results['test_recall'].mean()
            std_recall = cv_results['test_recall'].std() * 2  # 95% confidence interval
            
            mean_specificity = cv_results['test_specificity'].mean()
            std_specificity = cv_results['test_specificity'].std() * 2  # 95% confidence interval
            
            mean_f1_score = cv_results['test_f1_score'].mean()
            std_f1_score = cv_results['test_f1_score'].std() * 2  # 95% confidence interval            
            #--------------------------------------
            # Print the mean accuracy and recall with 95% confidence intervals
            print("Mean accuracy: %0.2f (± %0.2f)" % (mean_accuracy, std_accuracy))
            print("Mean recall (Sensitivity): %0.2f (± %0.2f)" % (mean_recall, std_recall))
            #----------------This is for simple accuracy matrics----------------------

            # # Perform 5-fold cross validation
            # scores = cross_val_score(model, X_train, y_train, cv=num_of_k_fold)

            # # Print the accuracy for each fold
            # print("Accuracy for each fold: ", scores)

            # # Print the mean accuracy and the 95% confidence interval of the score estimate
            # print("Mean accuracy: %0.2f (± %0.2f)" % (scores.mean(), scores.std() * 2))
            #--------------------------------------
            model.fit(X_train, y_train)
            # Predict the labels for the test set
            y_pred = model.predict(X_test)

            # Compute the accuracy (or any other relevant metric) on the test set
            test_accuracy = accuracy_score(y_test, y_pred)
            print("Test set accuracy: ", test_accuracy)
            print("HB TYPE: ", HB_TYPE)

            val_metrics_mean[HB_TYPE] = [mean_accuracy, mean_recall, mean_specificity, mean_f1_score]
            val_metrics_std[HB_TYPE] = [std_accuracy, std_recall, std_specificity, std_f1_score]
            HB_TYPE_accuraies[HB_TYPE].append(test_accuracy)
            accuracies[HB_TYPE] = HB_TYPE_accuraies[HB_TYPE]
            HB_TYPE_y_pred_and_y_test[HB_TYPE].append([y_pred, y_test])
        
        save_result={}
        save_result['val_metrics_mean'] = val_metrics_mean
        save_result['val_metrics_std'] = val_metrics_std
        save_result['accuracies'] = accuracies
        save_result['model_accuraies'] = HB_TYPE_accuraies
        save_result['current_time_seed'] = current_time_seed
        save_result['num_time'] = num_time
        save_result['HB_TYPE_y_pred_and_y_test'] = HB_TYPE_y_pred_and_y_test

        res[f'{num_time}'] = save_result
        print(res)
    np.save(output_fold + f'/{name}_result.npy', res)
