import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, LeaveOneOut
from sklearn.svm import SVC, SVR
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, confusion_matrix, make_scorer, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
CV = StratifiedKFold(n_splits=5)
# CV = LeaveOneOut() # StratifiedKFold(n_splits=5)
# 


scoring_scorer = make_scorer(f1_score, pos_label=1)


def get_class_weight_dict(labels):

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weight_dict= {i : class_weights[i] for i in range(len(class_weights))}
    
    print("Class weights: ", class_weight_dict)
    return class_weight_dict

def tune_random_forest(data, labels, weight_0=None):
    # Define the parameter grid for RandomForestClassifier
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Compute class weights
    if weight_0 is not None:
        class_weight_dict = {0: weight_0, 1: 1 - weight_0}
    else:
        class_weight_dict = {0: 1 - 0.99999999, 1: 0.99999999}
        
    # Create the GridSearchCV object
    grid_search = GridSearchCV(RandomForestClassifier(class_weight=class_weight_dict), 
                               param_grid, scoring=scoring_scorer, cv=5, verbose=1, n_jobs=20)

    # Fit the model
    grid_search.fit(data, labels)

    # Print the best parameters and the corresponding score
    print("Best parameters for Random Forest: ", grid_search.best_params_)
    print("Best Accuracy score: ", grid_search.best_score_)

    # Return the best estimator
    return grid_search.best_estimator_

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
    # class_weight_dict = get_class_weight_dict(labels)
    weight_1 = 0.925
    class_weight_dict = {0: 1 - weight_1, 1: weight_1}
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

def tune_knn(data, labels):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring=scoring_scorer, cv=5, verbose=1, n_jobs=12)
    grid_search.fit(data, labels)
    print("Best parameters for KNN: ", grid_search.best_params_)
    print("Best Sensitivity score for KNN: ", grid_search.best_score_)
    return grid_search.best_estimator_

def tune_mlp(data, labels):
    # Define the parameter grid for MLPClassifier
    class_weights = {0: 0.2, 1: 0.8}
    param_grid = {
        'hidden_layer_sizes': [(32, 32, 16), (64, 64, 32)],
        'activation': ['relu'],
        'solver': ['sgd', 'adam'],
        'learning_rate_init': [0.001, 0.01, 0.1]
    }
    # param_grid = {'activation': ['relu'], 'hidden_layer_sizes': [(32, 32, 16)], 'learning_rate_init': [0.1], 'solver': ['adam']}
    # Create the GridSearchCV object
    grid_search = GridSearchCV(MLPClassifier(max_iter=2000), param_grid, scoring=scoring_scorer, cv=5, verbose=1, n_jobs=15)


    # Fit the model
    grid_search.fit(data, labels)

    # Print the best parameters and the corresponding score
    print("Best parameters for MLP: ", grid_search.best_params_)
    print("Best Accuracy score: ", grid_search.best_score_)

    # Return the best estimator
    return grid_search.best_estimator_


def tune_gaussian_nb(data, labels, weight_0=None):
    # Define the parameter grid
    param_grid = {
        'var_smoothing': np.logspace(-9, 0, 10)
    }
    if weight_0 is None:
        weight_0 = 0.98
        class_prior = [weight_0, 1-weight_0]  # Increase the prior probability of the negative class
    else:
        class_prior = [weight_0, 1-weight_0]
    # Create the GaussianNB model
    gnb = GaussianNB(priors=class_prior) # priors=class_prior
    
    # Define the GridSearchCV
    grid_search = GridSearchCV(estimator=gnb, param_grid=param_grid, scoring=scoring_scorer, cv=5, verbose=1, n_jobs=20)
    
    # Fit the model
    grid_search.fit(data, labels)
    
    # Print the best parameters and score
    print("Best parameters for GaussianNB: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)
    
    # Return the best estimator
    return grid_search.best_estimator_

def tune_qda(data, labels, weight_0=None):
    # Define the parameter grid
    param_grid = {
        'reg_param': np.linspace(0, 1, 10)
    }

    # Define class weights

    # Compute sample weights
    # sample_weights = compute_sample_weight(class_weight=class_weights, y=labels)
    if weight_0 is  not None:
        priors_ = [weight_0, 1-weight_0]
    else:
        weight_0 = 0.8
        priors_ = [weight_0, 1-weight_0]
    qda = QuadraticDiscriminantAnalysis(priors=priors_)


    # Define the GridSearchCV
    grid_search = GridSearchCV(estimator=qda, param_grid=param_grid, scoring=scoring_scorer, cv=5, verbose=1, n_jobs=12)

    # Fit the model with sample weights
    grid_search.fit(data, labels)

    # Print the best parameters and score
    print("Best parameters for QDA: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    # Return the best estimator
    return grid_search.best_estimator_

def tune_lda(data, labels, weight_0=None):
    # Define the parameter grid
    param_grid = {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': [None, 'auto', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Only applicable if solver is 'lsqr' or 'eigen'
    }

    # Define class weights
    if weight_0 is not None:
        class_weights = [weight_0, 1-weight_0]

    else:
        class_weights = [0.8, 0.2]


    # Create the LDA model
    lda = LinearDiscriminantAnalysis(priors=class_weights)


    # Define the GridSearchCV
    grid_search = GridSearchCV(estimator=lda, param_grid=param_grid, scoring=scoring_scorer, cv=5, verbose=1, n_jobs=12)

    # Fit the model with sample weights
    grid_search.fit(data, labels)

    # Print the best parameters and score
    print("Best parameters for LDA: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    # Return the best estimator
    return grid_search.best_estimator_


def tune_sgd_classifier(data, labels, weight_0=None):
    # Define the parameter grid for SGDClassifier
    param_grid = {
        'loss': ['log_loss'],  # Using log loss to enable predict_proba
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'max_iter': [1000, 2000, 3000],
        'tol': [1e-4, 1e-3, 1e-2]
    }

    # Define class weights
    if weight_0 is not None:
        class_weight_dict = {0:weight_0, 1:1-weight_0}

    else:
        class_weight_dict = {0:0.2, 1:0.8}


    # Create the GridSearchCV object
    grid_search = GridSearchCV(SGDClassifier(class_weight=class_weight_dict), 
                               param_grid, scoring=scoring_scorer, cv=5, verbose=1, n_jobs=20)

    # Fit the model
    grid_search.fit(data, labels)

    # Print the best parameters and the corresponding score
    print("Best parameters for SGD Classifier: ", grid_search.best_params_)
    print("Best Accuracy score: ", grid_search.best_score_)

    # Return the best estimator
    return grid_search.best_estimator_

def define_classifier_for_classification_for_response(data, labels, weight_0=None):
    if weight_0 is not None:
        weight_0_name = str(weight_0).replace('.', '_')
    else:
        weight_0_name = ''
    # Define the classifiers dictionary including the Voting Classifier
    classifiers = {
        # "MLP": tune_mlp(data, labels),
        # "KNN": tune_knn(data, labels),
        "SVM": tune_svm(data, labels),  #SVC(class_weight=get_class_weight_dict(labels), kernel='rbf', C=10, gamma='auto', probability=True) # SVC(C=1000, coef0=0.5, degree=2, gamma='scale', kernel='poly', probability=True) 
        "XGBoost": tune_xgboost(data, labels),#XGBClassifier(scale_pos_weight=1e6),
        "Naive Bayes": tune_gaussian_nb(data, labels),
        "Random Forest": tune_random_forest(data, labels),
        "Discriminant Analysis(QDA)"+weight_0_name: tune_qda(data, labels),
        "Discriminant Analysis(LDA)"+weight_0_name: tune_lda(data, labels, weight_0=0.5),
        "SGDClassifier"+weight_0_name: tune_sgd_classifier(data, labels, weight_0=0.2),
    }
    
    return classifiers


def define_classifier_for_classification_for_partial_response(data, labels, weight_0=None):
    if weight_0 is not None:
        weight_0_name = str(weight_0).replace('.', '_')
    else:
        weight_0_name = ''
    # Define the classifiers dictionary including the Voting Classifier
    classifiers = {
        # "MLP": tune_mlp(data, labels),
        # "KNN": tune_knn(data, labels),
        # "SVM": tune_svm(data, labels),  #SVC(class_weight=get_class_weight_dict(labels), kernel='rbf', C=10, gamma='auto', probability=True) # SVC(C=1000, coef0=0.5, degree=2, gamma='scale', kernel='poly', probability=True) 
        # "XGBoost": tune_xgboost(data, labels),#XGBClassifier(scale_pos_weight=1e6),
        # "Naive Bayes"+weight_0_name: tune_gaussian_nb(data, labels, weight_0=weight_0),
        "Random Forest"+weight_0_name: tune_random_forest(data, labels, weight_0),
        # "Discriminant Analysis(LDA)"+weight_0_name: tune_lda(data, labels, weight_0=0.5),
        # "SGDClassifier"+weight_0_name: tune_sgd_classifier(data, labels, weight_0=0.52),
        # "Discriminant Analysis(QDA)"+weight_0_name: tune_qda(data, labels),
        
    }
    
    return classifiers