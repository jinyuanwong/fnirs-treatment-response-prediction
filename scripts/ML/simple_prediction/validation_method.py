import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

from sklearn.metrics import recall_score, roc_curve, roc_auc_score, accuracy_score, confusion_matrix, make_scorer, f1_score, auc


def calculate_metrics(y_true, y_pred, y_pred_prob):
    # roc_auc = roc_auc_score(y_true, y_pred)
    # accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc = roc_auc_score(y_true, y_pred_prob)
    specificity = tn / (tn + fp)
    accuracy = (sensitivity + specificity) / 2
    return {'balanced accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity, 'auc': auc}

def averaging_multiple_folds_result_into_one(predictions, num_k_folds, is_prob=False):
    predictions = np.array(predictions)
    predictions = np.reshape(predictions, (-1, num_k_folds))
    predictions = np.mean(predictions, axis=1)
    if not is_prob:
        for i in range(len(predictions)):
            predictions[i] = 1 if predictions[i] >= 0.5 else 0
    return predictions

def stratified_5_fold_classification(data, labels, classifiers):
    # Define the cross-validation strategy
    cv = StratifiedKFold(n_splits=5)

    # Initialize a dictionary to store the performance metrics
    metrics = ['roc_auc', 'balanced accuracy', 'sensitivity', 'specificity']
    model_performance = {name: {metric: [] for metric in metrics} for name in classifiers}
    result_table = []

    # Perform classification using each classifier
    for train_index, test_index in cv.split(data, labels):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            model_performance[name] = calculate_metrics(y_test, predictions)
            
    # Calculate and print average performance metrics
    print("\nAverage performance:")
    for name, metrics in model_performance.items():
        averages = {metric: np.mean(scores) for metric, scores in metrics.items()}
        result_table.append([name, averages['roc_auc'], averages['balanced accuracy'], averages['sensitivity'], averages['specificity']])
        
    # Print the results in Markdown table format
    print("\n## Model Performance")
    print("| Classifier | Average ROC AUC | Average bAcc | Average Sensitivity | Average Specificity |")
    print("|------------|-----------------|------------------|---------------------|---------------------|")
    for row in result_table:
        print(f"| {row[0]} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} | {row[4]:.4f} |")

def loocv_classification(data, labels, classifiers):
    # Define the cross-validation strategy
    cv = LeaveOneOut()
    # Initialize a dictionary to store the performance metrics
    metrics = ['roc_auc', 'balanced accuracy', 'sensitivity', 'specificity']
    model_performance = {name: {metric: [] for metric in metrics} for name in classifiers}
    all_outer_predictions = {name: [] for name in classifiers}
    result_table = []

    # Perform classification using each classifier
    for train_index, test_index in cv.split(data, labels):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)[0]
            all_outer_predictions[name].append(predictions)
    
    for name, predictions in all_outer_predictions.items():
        predictions = np.array(predictions)
        outer_metrics = calculate_metrics(labels, predictions)
        
        for metric, score in outer_metrics.items():
            model_performance[name][metric].append(score)
        
        averages = {metric: np.mean(scores) for metric, scores in model_performance[name].items()}
        result_table.append([name, averages['roc_auc'], averages['balanced accuracy'], averages['sensitivity'], averages['specificity']])
            
    # # Calculate metrics
    # roc_auc = roc_auc_score(y_test, predictions)
    # accuracy = accuracy_score(y_test, predictions)
    # sensitivity = recall_score(y_test, predictions)
    # tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    # specificity = tn / (tn + fp)
    # accuracy = (sensitivity + specificity) / 2
    # # Store metrics
    # model_performance[name]['roc_auc'].append(roc_auc)
    # model_performance[name]['balanced accuracy'].append(accuracy)
    # model_performance[name]['sensitivity'].append(sensitivity)
    # model_performance[name]['specificity'].append(specificity)
    
    # # Calculate and print average performance metrics
    # print("\nAverage performance:")
    # for name, metrics in model_performance.items():
    #     averages = {metric: np.mean(scores) for metric, scores in metrics.items()}
    #     result_table.append([name, averages['roc_auc'], averages['balanced accuracy'], averages['sensitivity'], averages['specificity']])
        
    # Print the results in Markdown table format
    print("\n## Model Performance")
    print("| Classifier | Average ROC AUC | Average bAcc | Average Sensitivity | Average Specificity |")
    print("|------------|-----------------|------------------|---------------------|---------------------|")
    for row in result_table:
        print(f"| {row[0]} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} | {row[4]:.4f} |")


def calculate_threshold_of_prediction(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def pred_using_threshold(y_pred_prob, threshold):
    return (y_pred_prob >= threshold).astype(int)

def nested_cross_validation_classification(data, labels, classifiers):
    outer_cv = LeaveOneOut()
    outer_metrics = ['balanced accuracy', 'sensitivity', 'specificity', 'auc']
    outer_performance = {name: {metric: [] for metric in outer_metrics} for name in classifiers}
    inner_performance = {name: {metric: [] for metric in outer_metrics} for name in classifiers}
    all_true_labels = []
    all_outer_predictions = {name: [] for name in classifiers}
    result_table = []

    num_k_folds = 5
    for train_index, test_index in outer_cv.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        all_true_labels.append(y_test[0])

        inner_cv = StratifiedKFold(n_splits=num_k_folds)
        
        for name, clf in classifiers.items():
            inner_scores = {metric: [] for metric in outer_metrics}
            
            for inner_train_index, inner_test_index in inner_cv.split(X_train, y_train):
                X_inner_train, X_inner_test = X_train[inner_train_index], X_train[inner_test_index]
                y_inner_train, y_inner_test = y_train[inner_train_index], y_train[inner_test_index]
                clf.fit(X_inner_train, y_inner_train)
                y_pred_prob = clf.predict_proba(X_inner_test)[:, 1]
                # optimal_threshold = calculate_threshold_of_prediction(y_inner_test, y_pred_prob)
                # y_pred = pred_using_threshold(y_pred_prob, optimal_threshold)
                y_pred = clf.predict(X_inner_test)
                inner_metrics = calculate_metrics(y_inner_test, y_pred, y_pred_prob)

                for metric, score in inner_metrics.items():
                    inner_scores[metric].append(score)
                outer_prediction = clf.predict(X_test)[0]
                outer_prediction_prob = clf.predict_proba(X_test)[0]
                # outer_prediction = (outer_prediction_prob[1] > optimal_threshold).astype(int)
                
                ## do not modify the order of the following two lines
                all_outer_predictions[name].append(outer_prediction) # 
                all_outer_predictions[name].append(outer_prediction_prob[1])
                
            inner_averages = {metric: np.mean(scores) for metric, scores in inner_scores.items()}
            for metric, avg in inner_averages.items():
                inner_performance[name][metric].append(avg)
    
    all_true_labels = np.array(all_true_labels)
    
    for name, predictions in all_outer_predictions.items():
        predictions = np.array(predictions)
        pred = predictions[::2]
        pred_prob = predictions[1::2]
        print(f"pred -> {pred}")
        print(f"pred_prob -> {pred_prob}")
        pred = averaging_multiple_folds_result_into_one(pred, num_k_folds)
        pred_prob = averaging_multiple_folds_result_into_one(pred_prob, num_k_folds, is_prob=True)
        
        outer_metrics = calculate_metrics(all_true_labels, pred, pred_prob)

        outer_performance[name] = outer_metrics
        outer_performance[name]['y_pred_test'] = pred_prob
        outer_performance[name]['y_true_test'] = all_true_labels

    # print("\n## Inner Cross-Validation Performance")
    # print("| Classifier | Average bAcc | Average Sensitivity | Average Specificity | Average AUC |")
    # print("|------------|-----------------|------------------|---------------------|---------------------|")
    for name, metrics in inner_performance.items():
        inner_performance[name] = avg_metrics = {metric: np.mean(scores) for metric, scores in metrics.items()}
    #     print(f"| {name} | {avg_metrics['balanced accuracy']:.4f} | {avg_metrics['sensitivity']:.4f} | {avg_metrics['specificity']:.4f} | {avg_metrics['auc']:.4f} |")
    
    # print("\n## Outer Cross-Validation Performance")
    # print("| Classifier | Average bAcc | Average Sensitivity | Average Specificity | Average AUC |")
    # print("|------------|-----------------|------------------|---------------------|---------------------|")
    # for name, metrics in outer_performance.items():
    #     print(f"| {name} | {metrics['balanced accuracy']:.4f} | {metrics['sensitivity']:.4f} | {metrics['specificity']:.4f} | {metrics['auc']:.4f} |")
    
    return inner_performance, outer_performance