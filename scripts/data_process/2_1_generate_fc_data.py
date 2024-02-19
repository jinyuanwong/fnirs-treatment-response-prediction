"""
Objective: 

2_0_generate_fc_data.py produces a data shape of (shape, 52, 52, 3) as data.npy in DMFC

This code is to seperate the data into 3 different data.npy files, each of them has a shape of (shape, 52, 52) in 
hbo hbr and hbt fold.




"""

import numpy as np
from scipy import stats
import os
import sys


two_kind_of_dataset = ['pre', 'pre_post']


three_fold = ['hbo', 'hbr', 'hbt']

def generate_fold(path):
    if not os.path.exists(path):
        os.makedirs(path)
        


def main():
    for index, hb_fold in enumerate(three_fold):
        generate_fold(f'allData/prognosis/DMFC/{hb_fold}/pre_post_treatment_hamd_reduction_50')
        generate_fold(f'allData/prognosis/DMFC/{hb_fold}/pre_treatment_hamd_reduction_50')
        for dataset in two_kind_of_dataset:
            data = np.load(f'allData/prognosis/DMFC/{dataset}_treatment_hamd_reduction_50/data.npy')
            label = np.load(f'allData/prognosis/DMFC/{dataset}_treatment_hamd_reduction_50/label.npy')
            if dataset == 'pre':
                hb = data[:,:,:,index]
            elif dataset == 'pre_post':
                hb = data[:,:,:,index,:]
            np.save(f'allData/prognosis/DMFC/{hb_fold}/{dataset}_treatment_hamd_reduction_50/data.npy', hb)
            np.save(f'allData/prognosis/DMFC/{hb_fold}/{dataset}_treatment_hamd_reduction_50/label.npy', label)
            
            
        

if __name__ == '__main__':
    main()
