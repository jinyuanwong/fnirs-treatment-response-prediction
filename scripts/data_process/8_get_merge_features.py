import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import os 
import sys
sys.path.append('/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning')

from scripts.plot.statistics.Complete.Regression_Distribution import *
from utils.fnirs_utils import normalize 
from utils.fnirs_utils import normalize_demographic
from utils.fnirs_utils import avg_every_ten_point_in_last_dimension

# time point 
data_timepoint = np.load('/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/allData/prognosis/pre_treatment_hamd_reduction_50/hb_data.npy')
hbo = data_timepoint[:,:,:1250]
hbo = avg_every_ten_point_in_last_dimension(hbo)
hbr = data_timepoint[:,:,1252:]
hbr = avg_every_ten_point_in_last_dimension(hbr)
hbt = hbo + hbr
hb = np.concatenate((hbo, hbr), axis=2)
hb = np.concatenate((hb, hbt), axis=2)
data_timepoint = hb
label_timepoint = np.load('/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/allData/prognosis/pre_treatment_hamd_reduction_50/label.npy')

# functional connectivity
data_fc = np.load('/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/allData/prognosis/DMFC/pre_treatment_hamd_reduction_50/data.npy')
data_fc = data_fc.reshape(data_fc.shape[0], data_fc.shape[1], -1)
data_fc = normalize(data_fc)
label_fc = np.load('/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/allData/prognosis/DMFC/pre_treatment_hamd_reduction_50/label.npy')

# selected features
data_sf = np.load('/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/allData/prognosis/pretreatment_benchmarks/yu_gnn/data.npy')
data_sf = np.nan_to_num(data_sf)
label_sf = np.load('/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/allData/prognosis/pretreatment_benchmarks/yu_gnn/label.npy')

# demographic features 
data_df = np.load('/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/allData/prognosis/pre_treatment_hamd_reduction_50/demographic_data.npy')
data_df = normalize_demographic(data_df)
new_data_df = np.zeros((data_df.shape[0], 52, data_df.shape[1]))
for i in range(new_data_df.shape[1]):
    new_data_df[:,i,:] = data_df
data_df = new_data_df
label_df = np.load('/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/allData/prognosis/pre_treatment_hamd_reduction_50/label.npy')

# print(label_timepoint == label_fc)
# print(label_timepoint == label_sf)

print('np.mean(data_timepoint) = ', np.mean(data_timepoint), ', shape -> ', data_timepoint.shape)
print('np.mean(data_fc) = ', np.mean(data_fc), ', shape -> ', data_fc.shape)
print('np.mean(data_sf) = ', np.mean(data_sf), ', shape -> ', data_sf.shape)
print('np.mean(data_df) = ', np.mean(data_df), ', shape -> ', data_df.shape)

merge_feature = np.concatenate((data_timepoint, data_fc, data_sf, data_df), axis=2)
print(merge_feature.shape)

np.save('/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/allData/prognosis/pre_treatment_hamd_reduction_50/merge_feature.npy', merge_feature) 
