import numpy as np 
from scipy.io import loadmat

data_fold = 'Prerequisite/data_all_original'

Hb_type = ['HbO', 'HbR', 'HbT']

# test one 

data_pth = '/Users/shanxiafeng/Documents/Code/python/fnirs_DL/JinyuanWang_pythonCode/Prerequisite/data_all_original/Fabeha_s Data/All 52-channel/all52CH_prep_HbO.mat'
data = loadmat(data_pth)

print('data')
