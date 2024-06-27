import numpy as np

def select_data(data: np.array, label, index):
    retain_arr = [i for i in range(data.shape[0]) if i not in index]
    return data[retain_arr], label[retain_arr]