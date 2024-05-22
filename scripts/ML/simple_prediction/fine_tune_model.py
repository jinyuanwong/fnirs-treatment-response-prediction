import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, LeaveOneOut
from sklearn.svm import SVC, SVR
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, confusion_matrix, make_scorer, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB


CV = StratifiedKFold(n_splits=5)
# CV = LeaveOneOut() # StratifiedKFold(n_splits=5)
# 


scoring_scorer = make_scorer(f1_score, pos_label=1)

def get_class_weight_dict(labels):

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weight_dict= {i : class_weights[i] for i in range(len(class_weights))}
    return class_weight_dict

def tune_xgboost(data, labels):
    # Define the parameter grid for XGBoost
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
    }

    # Create a StratifiedKFold object

    # Create the XGBClassifier object
    xgb = XGBClassifier(scale_pos_weight=1e6)

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring=scoring_scorer, cv=CV, verbose=1, n_jobs=12)

    # Fit the model
    grid_search.fit(data, labels)

    # Print the best parameters and the corresponding score
    print("Best parameters for XGBoost: ", grid_search.best_params_)
    print("Best Sensitivity score for XGBoost: ", grid_search.best_score_)

    # Return the best estimator
    return grid_search.best_estimator_

def tune_svm(data, labels):
    # Define the parameter grid for SVM
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'degree': [2, 3, 4],  # Applicable for 'poly' kernel
        'coef0': [0.0, 0.1, 0.5, 1.0],  # Applicable for 'poly' and 'sigmoid' kernels
        'probability': [True]
    }

    # Create a StratifiedKFold object

    # Create the SVC object
    class_weight_dict = get_class_weight_dict(labels)
    svc = SVC(class_weight=class_weight_dict)

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring=scoring_scorer, cv=CV, verbose=1, n_jobs=12)

    # Fit the model
    grid_search.fit(data, labels)

    # Print the best parameters and the corresponding score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best Accuracy score: ", grid_search.best_score_)

    # Return the best estimator
    return grid_search.best_estimator_


def tune_svr(data, labels):
    # Define the parameter grid for SVR
    param_grid = {
        'svr__C': [0.01, 0.1, 1, 10, 100],
        'svr__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'svr__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'svr__degree': [2, 3, 4],  # Applicable for 'poly' kernel
        'svr__coef0': [0.0, 0.1, 0.5, 1.0]  # Applicable for 'poly' and 'sigmoid' kernels
    }

    # Create a pipeline with StandardScaler and SVR
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=12)

    # Fit the model
    grid_search.fit(data, labels)

    # Print the best parameters and the corresponding score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best RMSE score: ", np.sqrt(-grid_search.best_score_))

    # Return the best estimator
    return grid_search.best_estimator_


def tune_gradient_boosting(data, labels):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, scoring=scoring_scorer, cv=5, verbose=1, n_jobs=12)
    grid_search.fit(data, labels)
    print("Best parameters for Gradient Boosting: ", grid_search.best_params_)
    return grid_search.best_estimator_

def tune_adaboost(data, labels):
    param_grid = {
        'n_estimators': [50, 100, 200, 500],
        'learning_rate': [0.01, 0.1, 1]
    }
    grid_search = GridSearchCV(AdaBoostClassifier(), param_grid, scoring=scoring_scorer, cv=5, verbose=1, n_jobs=12)
    grid_search.fit(data, labels)
    print("Best parameters for AdaBoost: ", grid_search.best_params_)
    return grid_search.best_estimator_


def tune_knn(data, labels):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring=scoring_scorer, cv=5, verbose=1, n_jobs=12)
    grid_search.fit(data, labels)
    print("Best parameters for KNN: ", grid_search.best_params_)
    print("Best Sensitivity score for KNN: ", grid_search.best_score_)
    return grid_search.best_estimator_

def tune_mlp(data, labels):
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (100,), (100, 50, 25)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
    }
    grid_search = GridSearchCV(MLPClassifier(max_iter=3000), param_grid, scoring=scoring_scorer, cv=5, verbose=1, n_jobs=20)
    grid_search.fit(data, labels)
    print("Best parameters for MLP: ", grid_search.best_params_)
    return grid_search.best_estimator_    

def tune_gaussian_nb(data, labels):
    # Define the parameter grid
    param_grid = {
        'var_smoothing': np.logspace(-9, 0, 10)
    }
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_prior = class_weights / class_weights.sum()  # Normalize to get prior probabilities
    
    # Create the GaussianNB model
    gnb = GaussianNB(priors=class_prior)
    
    # Define the GridSearchCV
    grid_search = GridSearchCV(estimator=gnb, param_grid=param_grid, scoring=scoring_scorer, cv=5, verbose=1, n_jobs=20)
    
    # Fit the model
    grid_search.fit(data, labels)
    
    # Print the best parameters and score
    print("Best parameters for GaussianNB: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)
    
    # Return the best estimator
    return grid_search.best_estimator_



def define_classifier_for_classification(data, labels):
    svm_clf = tune_svm(data, labels)#SVC(class_weight=get_class_weight_dict(labels), kernel='rbf', C=10, gamma='auto', probability=True)
    xgb_clf = tune_xgboost(data, labels)#XGBClassifier(scale_pos_weight=1e6)
    nb_clf = tune_gaussian_nb(data, labels)
    
    # Define the classifiers dictionary including the Voting Classifier
    classifiers = {
        "SVM": svm_clf,
        "XGBoost": xgb_clf,
        "Naive Bayes": nb_clf
    }
    return classifiers