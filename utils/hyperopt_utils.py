
from hyperopt import fmin, tpe, Trials, hp, STATUS_OK
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut
from xgboost import XGBClassifier
import numpy as np

from hyperopt import space_eval
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, recall_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import numpy as np

def skf_train(best_params):
    stratified_kfold = StratifiedKFold(n_splits=5)

    # Initialize the classifier with the best parameters
    clf = XGBClassifier(**best_params)
    # Lists to store metrics for each fold
    accuracies = []
    sensitivities = []
    specificities = []
    auc_scores = []

    # Perform the stratified 5-fold cross-validation
    for train_index, test_index in stratified_kfold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]  # probabilities for positive class

        # Calculate metrics
        acc = balanced_accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        sensitivity = recall_score(y_test, y_pred)  # Recall is the same as sensitivity
        sensitivities.append(sensitivity)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        specificities.append(specificity)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        auc_scores.append(auc)
        return accuracies, sensitivities, specificities, auc_scores
    
def loocv_train(best_params):
    # Initialize the classifier with the best parameters
    clf = XGBClassifier(**best_params)


    # Setup Leave-One-Out Cross-Validation
    loo = LeaveOneOut()

    # Arrays to store all predictions and true labels
    all_preds = []
    all_true_labels = []
    all_pred_probas = []

    # Perform the LOOCV
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Clone the classifier for each fold to ensure a clean slate
        clf_fold = clone(clf)

        # Train the model
        clf_fold.fit(X_train, y_train)

        # Make predictions
        y_pred = clf_fold.predict(X_test)
        y_pred_proba = clf_fold.predict_proba(X_test)[:, 1]  # probabilities for the positive class

        # Store predictions and true labels
        all_preds.extend(y_pred)
        all_true_labels.extend(y_test)
        all_pred_probas.extend(y_pred_proba)

    # Convert lists to arrays for metric calculation
    all_preds = np.array(all_preds)
    all_true_labels = np.array(all_true_labels)
    all_pred_probas = np.array(all_pred_probas)

    # Calculate overall metrics
    overall_acc = balanced_accuracy_score(all_true_labels, all_preds)
    overall_sensitivity = recall_score(all_true_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_true_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    overall_specificity = tn / (tn + fp)
    overall_auc = roc_auc_score(all_true_labels, all_pred_probas)
    return overall_acc, overall_sensitivity, overall_specificity, overall_auc



def get_best_hyperparameters(X, y, num_evals=200, random_seed=1024):
    # np.random.seed(random_seed)
    # Define the objective function
    def objective(params):
        # Initialize classifier
        clf = XGBClassifier(**params)
        
        # Setup StratifiedKFold
        skf = StratifiedKFold(n_splits=5)
        loo = LeaveOneOut()
        
        # List to store the AUC scores for each validation scheme
        auc_scores_skf = []
        recall_scores_skf = []    
        all_pred_probas = []
        
        # Stratified 5-Fold CV
        for train_index, test_index in skf.split(X, y):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
            
            clf_fold = clone(clf)
            clf_fold.fit(X_train_fold, y_train_fold)
            preds = clf_fold.predict_proba(X_test_fold)[:,1]
            fold_recall = f1_score(y_test_fold, preds>0.5)
            recall_scores_skf.append(fold_recall)

        # Leave-One-Out CV
        for train_index, test_index in loo.split(X):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]
            clf_fold = clone(clf)
            clf_fold.fit(X_train_fold, y_train_fold)
            preds = clf_fold.predict_proba(X_test_fold)[:,1]
            all_pred_probas.append(preds)
            
        all_pred_probas = np.array(all_pred_probas)    
        all_pred_labels = (all_pred_probas > 0.5).astype(int)
        loocv_recall = f1_score(y, all_pred_labels)


        mean_recall_skf = np.mean(recall_scores_skf) * 0.45
        mean_recall_loo = loocv_recall * 0.55

        combined_recall = mean_recall_skf + mean_recall_loo

        return {'loss': -combined_recall, 'status': STATUS_OK}


    # loocv_metrics = []
    # skf_metrics = []
    # Define the search space
    space = {
        'learning_rate': hp.loguniform('learning_rate', -6.907755278982137, 0), # loguniform for 0.001 to 1
        'min_child_weight': hp.uniform('min_child_weight', 0.8, 2.0),
        'max_depth': hp.choice('max_depth', range(3, 16)),
        'gamma': hp.uniform('gamma', 0.0, 2.0),
        'lambda': hp.uniform('lambda', 0.0, 4.0),
        'alpha': hp.uniform('alpha', 0.0, 2.0),
        'n_estimators': hp.choice('n_estimators', range(50, 301)),
        'scale_pos_weight': hp.loguniform('scale_pos_weight', np.log(1), np.log(1e8)),
    }


    # Setup Trials
    trials = Trials()

    # Run optimization
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=num_evals,
        trials=trials,
        rstate=np.random.default_rng(random_seed)
    )

    print("Best hyperparameters:", best)
    best_params = space_eval(space, best)
    # accuracies, sensitivities, specificities, auc_scores = skf_train(best_params)
    # overall_acc, overall_sensitivity, overall_specificity, overall_auc = loocv_train(best_params)
    # skf_metrics.append([accuracies, sensitivities, specificities, auc_scores])
    # loocv_metrics.append([overall_acc, overall_sensitivity, overall_specificity, overall_auc])
    return best_params

def get_best_hyperparameters_skf_inside_loocv_monitoring_recall_bacc(X, y, num_evals=200, random_seed=1024):
    # np.random.seed(random_seed)
    # Define the objective function
    def objective(params):
        # Initialize classifier
        clf = XGBClassifier(**params)
        
        # Setup StratifiedKFold
        skf = StratifiedKFold(n_splits=5)
        loo = LeaveOneOut()
        
        # List to store the AUC scores for each validation scheme
        auc_scores_skf = []
        recall_scores_skf = []    
        balanced_accuracy_scores_skf = []
        all_pred_probas = []
        


        # Leave-One-Out CV
        for train_index, test_index in loo.split(X):
            X_train_fold, X_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]            
            clf_fold = clone(clf)
            clf_fold.fit(X_train_fold, y_train_fold)
            preds = clf_fold.predict_proba(X_test_fold)[:,1]
            all_pred_probas.append(preds)
            # Stratified 5-Fold CV
            for train_index, val_index in skf.split(X_train_fold, y_train_fold):
                inner_X_train_fold, X_val_fold = X_train_fold[train_index], X_train_fold[val_index]
                inner_y_train_fold, y_val_fold = y_train_fold[train_index], y_train_fold[val_index]
                
                clf_fold = clone(clf)
                clf_fold.fit(inner_X_train_fold, inner_y_train_fold)
                preds = clf_fold.predict_proba(X_val_fold)[:,1]
                fold_recall = f1_score(y_val_fold, preds>0.5)
                recall_scores_skf.append(fold_recall)
                balanced_accuracy_scores_skf.append(balanced_accuracy_score(y_val_fold, preds>0.5))
            
        all_pred_probas = np.array(all_pred_probas)    
        all_pred_labels = (all_pred_probas > 0.5).astype(int)
        loocv_recall = f1_score(y, all_pred_labels)
        loocv_bacc = balanced_accuracy_score(y, all_pred_labels)


        mean_recall_skf = np.mean(recall_scores_skf) * 0.2
        mean_recall_loo = loocv_recall * 0.3
        
        mean_bacc_skf = np.mean(balanced_accuracy_scores_skf) * 0.2
        mean_bacc_loo = loocv_bacc * 0.3

        combined_recall = mean_recall_skf + mean_recall_loo + mean_bacc_skf + mean_bacc_loo

        return {'loss': -combined_recall, 'status': STATUS_OK}


    # loocv_metrics = []
    # skf_metrics = []
    # Define the search space
    space = {
        'learning_rate': hp.loguniform('learning_rate', -6.907755278982137, 0), # loguniform for 0.001 to 1
        'min_child_weight': hp.uniform('min_child_weight', 0.8, 2.0),
        'max_depth': hp.choice('max_depth', range(3, 16)),
        'gamma': hp.uniform('gamma', 0.0, 2.0),
        'lambda': hp.uniform('lambda', 0.0, 4.0),
        'alpha': hp.uniform('alpha', 0.0, 2.0),
        'n_estimators': hp.choice('n_estimators', range(50, 301)),
        'scale_pos_weight': hp.loguniform('scale_pos_weight', np.log(1), np.log(1e8)),
    }


    # Setup Trials
    trials = Trials()

    # Run optimization
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=num_evals,
        trials=trials,
        rstate=np.random.default_rng(random_seed)
    )

    print("Best hyperparameters:", best)
    best_params = space_eval(space, best)
    # accuracies, sensitivities, specificities, auc_scores = skf_train(best_params)
    # overall_acc, overall_sensitivity, overall_specificity, overall_auc = loocv_train(best_params)
    # skf_metrics.append([accuracies, sensitivities, specificities, auc_scores])
    # loocv_metrics.append([overall_acc, overall_sensitivity, overall_specificity, overall_auc])
    return best_params
