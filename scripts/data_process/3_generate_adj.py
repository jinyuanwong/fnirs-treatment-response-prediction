import scipy.sparse as sp
import numpy as np 
import matplotlib.pyplot as plt

def generate_fnirs_adj():
    matrix = sp.csr_matrix((52, 52), dtype=int)
    for i in range(10):
        if i > 0 and i < 9:
            matrix[i, i-1] = 1
            matrix[i, i+1] = 1
        elif i == 0:
            matrix[i, i+1] = 1
        else:
            matrix[i, i-1] = 1
        matrix[i, i+10] = 1
        matrix[i, i+11] = 1

    for i in range(10, 21):
        if i > 10 and i < 20:
            matrix[i, i-11] = 1
            matrix[i, i-10] = 1
            matrix[i, i-1] = 1
            matrix[i, i+1] = 1
            matrix[i, i+10] = 1
            matrix[i, i+11] = 1
        if i == 10:
            matrix[i, i-10] = 1
            matrix[i, i+1] = 1
            matrix[i, i+11] = 1
        if i == 20:
            matrix[i, i-11] = 1
            matrix[i, i-1] = 1
            matrix[i, i+10] = 1

    for i in range(21, 31):
        matrix[i, i-11] = 1
        matrix[i, i-10] = 1
        if i > 21 and i < 30:
            matrix[i, i-1] = 1
            matrix[i, i+1] = 1
        if i == 21:
            matrix[i, i+1] = 1

        if i == 30:
            matrix[i, i-1] = 1
        matrix[i, i+10] = 1
        matrix[i, i+11] = 1

    begin = 31
    end = 42

    for i in range(begin, end):
        if i > begin and i < end-1:
            matrix[i, i-11] = 1
            matrix[i, i-10] = 1
            matrix[i, i-1] = 1
            matrix[i, i+1] = 1
            matrix[i, i+10] = 1
            matrix[i, i+11] = 1
        if i == begin:
            matrix[i, i-10] = 1
            matrix[i, i+1] = 1
            matrix[i, i+11] = 1
        if i == end-1:
            matrix[i, i-11] = 1
            matrix[i, i-1] = 1
            matrix[i, i+10] = 1

    begin = 42
    end = 52

    for i in range(begin, end):
        if i > begin and i < end-1:
            matrix[i, i-11] = 1
            matrix[i, i-10] = 1
            matrix[i, i-1] = 1
            matrix[i, i+1] = 1
        if i == begin:
            matrix[i, i-11] = 1
            matrix[i, i-10] = 1
            matrix[i, i+1] = 1
        if i == end-1:
            matrix[i, i-11] = 1
            matrix[i, i-10] = 1
            matrix[i, i-1] = 1
    return matrix

adj = generate_fnirs_adj()
np_adj = adj.toarray()

all_folds = ['allData/prognosis/pre_post_treatment_hamd_reduction_50/',
             'allData/prognosis/pre_treatment_hamd_reduction_50/']
for fold in all_folds:
    label = np.load(fold + 'label.npy')
    print(label.shape)
    same_subject_of_adj = np.tile(np_adj, (label.shape[0],1,1))
    print(same_subject_of_adj.shape)
    # save to 
    np.save(fold+'adj_matrix', same_subject_of_adj)
