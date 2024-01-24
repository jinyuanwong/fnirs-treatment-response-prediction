import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, f1_score, roc_auc_score


datapath = '/Users/shanxiafeng/Documents/Code/python/fnirs_DL/JinyuanWang_pythonCode/allData/prognosis/data.npy'
labelpath= '/Users/shanxiafeng/Documents/Code/python/fnirs_DL/JinyuanWang_pythonCode/allData/prognosis/label.npy'

def SVM_RFECV(X_train, y_train,C=0.1, n=30):
    svm = SVC(kernel="linear", C=C)
    rfecv = RFECV(estimator=svm, step=1, cv=StratifiedKFold(5),
                    min_features_to_select=n, scoring='accuracy')
    rfecv.fit(X_train, y_train)
    best_features = rfecv.support_
    return best_features

def normalize_individual(data):
    # Iterate over each subject | optimized instead of using for
    normalized_data = np.empty_like(data)
    for i in range(data.shape[0]):
        # Calculate the mean and standard deviation for the current subject
        mean = np.mean(data[i])
        std = np.std(data[i])

        # Perform z-normalization for the current subject
        normalized_data[i] = (data[i] - mean) / std
        
    # mean = np.mean(data, axis=(1,2))
    # std = np.std(data, axis=(1,2))
    
    return normalized_data

def read_hb(datapath, labelpath):
    
    data = np.load(datapath)
    data = np.transpose(data, (0,2,1,3,4))

    label = np.load(labelpath)
    change = label[:,1] - label[:,0]

    hbo = data[:,:,:,0,:]
    hbr = data[:,:,:,1,:]

    hbo_base = normalize_individual(hbo[...,0])
    hbo_t8 = normalize_individual(hbo[...,1])


    hbr_base = normalize_individual(hbr[...,0])
    hbr_t8 = normalize_individual(hbr[...,1])
    return hbo_base, hbo_t8, hbr_base, hbr_t8

def compute_correlation(x, y, method='pearsonr'):
    if method == 'pearsonr':
        corr, _ = stats.pearsonr(x, y)
    else:
        # raise 
        raise ValueError('Method not supported')
    return corr

def compute_dmfc(data):
    dm_data = data
    feature_shape = dm_data.shape 
    dmfc = np.zeros((dm_data.shape[0], dm_data.shape[1], dm_data.shape[1]))
    for sub in range(feature_shape[0]):
        for ch_1 in range(feature_shape[1]):
            for ch_2 in range(feature_shape[1]):
                if ch_2 < ch_1: continue
                corr = compute_correlation(
                    dm_data[sub, ch_1],dm_data[sub, ch_2])
                dmfc[sub, ch_1, ch_2] = corr
                dmfc[sub, ch_2, ch_1] = corr
    return dmfc



def plotting_res(y_pred, y_test, mse, random_state, ax=None):
    if not ax:
        plt.figure(figsize=(10,10))
        
        plt.title(f'Random seed = {random_state}')

        plt.plot(y_pred, label=f'predict (mse={mse})', linestyle='--')
        plt.plot(y_test, label='true', linestyle='--')
        
        plt.scatter(list(range(len(y_pred))),y_pred)
        plt.scatter(list(range(len(y_test))),y_test)
        
        plt.legend()
    else: 
        ax.plot(y_pred, label=f'predict (mse={mse})', linestyle='--')
        ax.plot(y_test, label='true', linestyle='--')
        
        ax.scatter(list(range(len(y_pred))),y_pred)
        ax.scatter(list(range(len(y_test))),y_test)
        
        ax.legend()
def apply_model_to_predict_score(model, feature_2d, labels, alpha = 1.0, random_state=24, plotting=False, ax=None):
    X_train, X_test, y_train, y_test = train_test_split(feature_2d, labels, test_size=0.2, random_state=random_state)

    if model == 'Ridge':
        # Initialize Ridge regression model
        model = Ridge(alpha=alpha)  # Alpha is the regularization strength

    if model == 'ElasticNet':
        # Initialize ElasticNet model
        model = ElasticNet(alpha=alpha, l1_ratio=0.5)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    if plotting:
        plotting_res(y_pred, y_test, mse, random_state, ax=ax)


hbo_base, hbo_t8, hbr_base, hbr_t8 = read_hb(datapath, labelpath)
fc_hbo_base, fc_hbo_t8 = compute_dmfc(hbo_base), compute_dmfc(hbo_t8)
fc_hbr_base, fc_hbr_t8 = compute_dmfc(hbr_base), compute_dmfc(hbr_t8)

scores = np.load(labelpath)
score_change_percentage = (scores[:,1]-scores[:,0])/scores[:,0] # score changes

fc_hb_base = np.concatenate((fc_hbo_base.reshape((fc_hbo_base.shape[0], -1)), fc_hbr_base.reshape((fc_hbo_base.shape[0], -1))), axis=1)

score_change_percentage = (scores[:,1]-scores[:,0])/scores[:,0] # score changes
remission_label = np.zeros_like(score_change_percentage)
for i in range(len(score_change_percentage)):
    if score_change_percentage[i] < -0.25:
        remission_label[i] = 1
    else:
        remission_label[i] = 0
        
        
        
X = fc_hb_base
y = remission_label


loo = LeaveOneOut()
loocv_y = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Hello you!")
    best_features = SVM_RFECV(X_train, y_train)
    
    X_train_selected = X_train[:, best_features]
    X_test_selected = X_test[:, best_features]
    
    svm_final = SVC(kernel="linear")
    svm_final.fit(X_train_selected, y_train)
    y_pred = svm_final.predict(X_test_selected)
    
    loocv_y.append([y_test, y_pred])
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %.3f' % accuracy)
    
np.save('./loocv_y.npy', loocv_y)