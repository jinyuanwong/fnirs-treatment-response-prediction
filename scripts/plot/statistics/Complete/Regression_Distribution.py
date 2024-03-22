"""

plot the distribution of HAMD recovery 

Do regression prediction on the HAMD recovery

Permutation test 
- Randomly pair the labels and features, to see the result 

return: 

1. plot 
2. MSE score 
3. regression 
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import cross_val_predict
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet, Ridge, Lasso
from utils.fnirs_utils import avg_every_ten_point_in_last_dimension
from utils.fnirs_utils import specify_feature_extraction
from scripts.ML.Complete import TimeFeature_prognosis

def get_pretreatment_hamd_reduction_percentage(path='/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/prognosis/base_label.npy'):
    data = np.load(path)
    reduction_percent = (data[:,1] - data[:,0]) / data[:,0]
    return reduction_percent

    

# 1. plot the distribution of HAMD recovery


def plot_hamd_distribution():
    hamd_data = get_pretreatment_hamd_reduction_percentage()
    sns.set_style('whitegrid')

    # Plotting the histogram with density instead of count on y-axis
    sns.histplot(hamd_data, bins=30, kde=False, stat="count", label='HAMD Recovery Histogram')

    # Calculate the mean and standard deviation, which are needed to plot the normalization curve
    mu, std = norm.fit(hamd_data)

    # Plot the normalization curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Normalization Curve')

    title = "Fit results: μ = %.2f,  σ = %.2f" % (mu, std)
    plt.title(title)

    plt.legend()
    plt.show()
    # hamd_data = get_pretreatment_hamd_reduction_percentage()
    # plt.hist(hamd_data, bins=20, alpha=0.7, color='blue')
    # plt.title('Distribution of HAMD recovery')
    # plt.xlabel('HAMD recovery')
    # plt.ylabel('Frequency')
    # plt.show()

# plot_hamd_distribution()


def plot_predict_true(label, predicted, title_name):
    
    plt.scatter(label, predicted)

    r, p = pearsonr(label, predicted)
    print(f"Correlation coefficient (r): {r}")
    print(f"P-value: {p}")
    plt.plot([-1, 1], [-1, 1], color='red', linestyle='-')
    # plt.xlim([-1.3, 1.3])
    # plt.ylim([-1.3, 1.3])
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title('Comparison of Predicted and True Values r = {:.4f}, p = {:.4f}, {}'.format(r, p, title_name))
    plt.show()
    
    

def CV_predict(model, data, label, k_fold=5):
    n_samples = len(label)
    n_samples_per_fold = n_samples // k_fold
    a = 1
    all_pred = []
    for k in range(k_fold):
        train_data = np.concatenate([data[:k*n_samples_per_fold], data[(k+1)*n_samples_per_fold:]], axis=0)
        test_data = data[k*n_samples_per_fold:(k+1)*n_samples_per_fold]
        train_label = np.concatenate([label[:k*n_samples_per_fold], label[(k+1)*n_samples_per_fold:]], axis=0)
        test_label = label[k*n_samples_per_fold:(k+1)*n_samples_per_fold]
        
        clf = model
        clf.fit(train_data, train_label)
        pred = clf.predict(test_data)
        all_pred.append(pred)
    all_pred = np.array(all_pred).flatten()
    return all_pred



def get_pretreatment_data_percentage():

    data = np.load('/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/prognosis/base_data.npy')
    data = data[..., 0]
    data = np.transpose(data, (0, 2, 1))
    data = data[...,:1250]
    data = avg_every_ten_point_in_last_dimension(data)
    data = specify_feature_extraction(data)
    
    return data

def get_pretreatment_hamd_reduction_percentage(path='/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/prognosis/base_label.npy'):
    data = np.load(path)
    reduction_percent = (data[:,1] - data[:,0]) / data[:,0]
    return reduction_percent





def retrieve_best_indice_from_4_yu_time_feature():
    _, _, model = TimeFeature_prognosis.predict_based_on_automl()

    # Assuming model.feature_importances_ is a 1D array
    indices = np.argsort(model.feature_importances_)[-4:]
    return indices
    
    
def plot_regression():
    data = get_pretreatment_data_percentage()
    print('1 datashape:', data.shape)
    label = get_pretreatment_hamd_reduction_percentage()
    
    # used the discriminative features
    indices = retrieve_best_indice_from_4_yu_time_feature()
    data = data[:, indices]
    print('data shapa:', data.shape)
    
    
    model = Ridge()
    model_name = 'Ridge'
    cv = 5
    
    predicted = cross_val_predict(model, data, label, cv=cv)
    plot_predict_true(label, predicted, model_name+f' cv={cv}')
    
    
    
def find_best_model():

    data = get_pretreatment_data_percentage()
    label = get_pretreatment_hamd_reduction_percentage()
    
    # used the discriminative features
    indices = retrieve_best_indice_from_4_yu_time_feature()
    data = data[:, indices]
    model_name_dicit = {
        'LinearRegression': LinearRegression(),
        # 'Lasso': Lasso(),
        # 'ElasticNet': ElasticNet(),
        # 'Bayesian Ridge Regression': BayesianRidge(),
        'Ridge Regression': Ridge(),
    }
    for cv in [5, 65]:
        for model_name, model in model_name_dicit.items():
            # predicted = CV_predict(model, data, label, k_fold=cv)
            predicted = cross_val_predict(model, data, label, cv=5)
            plt.figure()
            plot_predict_true(label, predicted, model_name+f' cv={cv}')