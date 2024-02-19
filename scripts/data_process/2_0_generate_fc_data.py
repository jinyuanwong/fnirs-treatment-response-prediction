"""
Objective: Generate functional connectivity data from 

1. pre_post_treatment_hamd_reduction_50/data.npy
2. pre_treatment_hamd_reduction_50/data.npy

notes: the data is normalized within each individual subject.

Based on pearsonr correlation coefficient

output: 

For baseline hemolgobin concentration data to baseline functinal connectivity data
- data.npy (Subject, Channel, Channel, 3[HbO, HbR, HbT])
    - The last dimension is the different hemoglobin type of functional connectivity.
    
For T0 and T8 hemolgobin concentration data to T0_T8 functinal connectivity data

- data.npy (Subject, Channel, Channel, 3[HbO, HbR, HbT], 2[pre, post])
    - The last dimension is the different hemoglobin type of functional connectivity.
    - The last dimension is the different time point of functional connectivity.


"""

import numpy as np
from scipy import stats
import os
import sys


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
    if feature_shape[1] != 52:
        raise ValueError(
            'The shape of feature_shape should be (subject, 52, time)')
    else:
        print(
            "Nice, the feature_shape is correct, its shape[1] is 52 (channel)")

    dmfc = np.zeros((dm_data.shape[0], dm_data.shape[1], dm_data.shape[1]))
    for sub in range(feature_shape[0]):
        for ch_1 in range(feature_shape[1]):
            for ch_2 in range(feature_shape[1]):
                if ch_2 < ch_1:
                    continue
                corr = compute_correlation(
                    dm_data[sub, ch_1], dm_data[sub, ch_2])
                dmfc[sub, ch_1, ch_2] = corr
                dmfc[sub, ch_2, ch_1] = corr
    return dmfc


def read_data(path):
    data = np.load(path + '/data.npy')
    label = np.load(path + '/label.npy')
    return data, label


def save_data(path, data, label):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path+'/data.npy', data)
    np.save(path+'/label.npy', label)


"""
hb is a shape of (subject, channel, time)

time is 2502, 
    - :1251  is hbo 
    - 1251: is hbr
    
    
hbt is calculated by adding hbo and hbr
"""
def seperate_hb_compute_its_dmfc(hb):
    shape = hb.shape
    hbo = hb[..., :shape[-1]//2]
    hbr = hb[..., shape[-1]//2:]
    hbt = hbo+hbr
    fc_hbo = compute_dmfc(hbo)
    fc_hbr = compute_dmfc(hbr)
    fc_hbt = compute_dmfc(hbt)
    fc_hb = np.concatenate(
        (fc_hbo[..., np.newaxis], fc_hbr[..., np.newaxis], fc_hbt[..., np.newaxis]), axis=-1)

    return fc_hb


def compute_baseline_T8_dynamic_functional_connectivity(input_path, output_path):
    hb, label = read_data(input_path)

    # hb1 is the baseline time signal
    hb1 = hb[..., 0]
    # hb2 is the T8 time signal
    hb2 = hb[..., 1]

    fc_hb1 = seperate_hb_compute_its_dmfc(hb1)
    fc_hb2 = seperate_hb_compute_its_dmfc(hb2)

    fc_hb = np.concatenate(
        (fc_hb1[..., np.newaxis], fc_hb2[..., np.newaxis]), axis=-1)

    save_data(output_path, fc_hb, label)


def compute_baseline_dynamic_functional_connectivity(input_path, output_path):
    hb, label = read_data(input_path)
    fc_hb = seperate_hb_compute_its_dmfc(hb)
    save_data(output_path, fc_hb, label)


def main(timeline):
    print('You are using ', timeline)

    if timeline not in ['T0', 'T0_T8']:
        raise ValueError('timeline should be T0 or T0_T8')

    # For baseline hemolgobin concentration data to baseline functinal connectivity data
    if timeline == 'T0':
        baseline_input_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/prognosis/pre_treatment_hamd_reduction_50'
        dmfc_baseline_output_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/prognosis/DMFC/pre_treatment_hamd_reduction_50'
        compute_baseline_dynamic_functional_connectivity(
            baseline_input_path, dmfc_baseline_output_path)

    # For baseline_and_T8 time hemolgobin concentration  data to T0_T8 functinal connectivity data
    # the data shape is different
    if timeline == "T0_T8":
        baseline_and_T8_input_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/prognosis/pre_post_treatment_hamd_reduction_50'
        dmfc_baseline_and_T8_output_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction/allData/prognosis/DMFC/pre_post_treatment_hamd_reduction_50'
        compute_baseline_T8_dynamic_functional_connectivity(
            baseline_and_T8_input_path, dmfc_baseline_and_T8_output_path)


if __name__ == '__main__':
    arg = sys.argv
    main(arg[1])
