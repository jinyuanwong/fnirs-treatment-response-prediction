
"""

plt.title('Dynamic Functional Connectivity between responders and nonresponders', fontsize=15, fontweight='bold')


"""
# Remember to install chord-6.0.1
from chord import Chord
# load 
import sys
import time

from sklearn.model_selection import cross_val_score,train_test_split
from datetime import date
import numpy as np
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import pingouin as pg
import subprocess
import os 

output_fold = 'FigureTable/statics/DMFC'

if not os.path.exists(output_fold):
    os.makedirs(output_fold)
    

def zero_diagnonal(arr):
    # Loop over the first and last dimension
    for i in range(arr.shape[0]):  # Loop over subjects
        for j in range(arr.shape[-1]):  # Loop over views
            np.fill_diagonal(arr[i, :, :, j], 0)
    return arr


def show(adj, labels, name):
    adj = zero_diagnonal(adj)
    hc_adj = adj[np.where(labels==1)]
    md_adj = adj[np.where(labels==0)]
    count = 0

    num_view = adj.shape[-1]
    p_view = np.zeros((52,52,num_view))
    effect_size = np.zeros((52,52,num_view))
    for view in range(num_view):
        for seed in range(52):
            for target in range(52):
                hc_val = hc_adj[:, seed, target, view]
                md_val = md_adj[:, seed, target, view]

                stat, p1 = mannwhitneyu(hc_val,md_val)
                p_view[seed, target, view] = p1
                
                # Calculate Hedges' g
                effect_size[seed, target, view] = pg.compute_effsize(hc_val,md_val, eftype='Hedges')
                

    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.colors import LogNorm


    colors = ['red','white']
    # cmap = LinearSegmentedColormap.from_list('RedToWhite', colors)
    # colors = [(1, 0, 0), (1, 1, 1)]  # Red to White
    # colors = ['#FF0000', '#FF3333', '#FF6666', '#FF9999', '#FFCCCC', '#FFFFFF']
    # Create a colormap with the defined colors
    cmap = LinearSegmentedColormap.from_list('RedToWhite', colors, N=256)


    colors_effect_size = ['blue','white','red']
    # cmap = LinearSegmentedColormap.from_list('RedToWhite', colors)
    # colors = [(1, 0, 0), (1, 1, 1)]  # Red to White
    # colors = ['#FF0000', '#FF3333', '#FF6666', '#FF9999', '#FFCCCC', '#FFFFFF']
    # Create a colormap with the defined colors
    # Define the colors in hexadecimal format
    colors = ['#448196', 'white', '#c45c3d']

    # Create the custom color map
    cmap_effect_size = LinearSegmentedColormap.from_list('CustomColorMap', colors, N=256)

    # cmap_effect_size = LinearSegmentedColormap.from_list('BlueToWhiteToRed', colors_effect_size, N=256)

    # Set up a 2x4 grid for plotting
    fig, axes = plt.subplots(nrows=2, ncols=num_view, figsize=(12, 6))

    # Flatten axes array for easy indexing
    axes = axes.flatten()

    A_adj_name = ['HbO', 'HbR', 'HbT']

    # Loop through images and axes to display each image
    for idx, ax in enumerate(axes):
        if idx < num_view: 
            im=ax.imshow(p_view[:,:,idx], norm=LogNorm(vmin=0.001, vmax=0.05), cmap=cmap)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.05, ticks=[0.001, 0.01, 0.05])
            cbar.ax.set_yticklabels(['0.001', '0.01', '0.05'], fontsize=12, fontweight='bold')

            # ax.colorbar()
        else: 
            im=ax.imshow(effect_size[:,:,idx-num_view], vmin=-1.00, vmax=1.00, cmap=cmap_effect_size)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.05, ticks=[-1.00, -0.5, 0, 0.5, 1.00])
            cbar.ax.set_yticklabels(['-1.00', '-0.5', '0', '0.5', '1.00'], fontsize=12, fontweight='bold')

            # ax.colorbar()
        # ax.axis('off')  # Turn off axis
        ax.set_title(f'{A_adj_name[idx%num_view]}',fontsize=13, fontweight='bold')  # Set title for each subplot
            # Set axis ticks
        ax.set_xticks([0, 25, 51])
        ax.set_yticks([0, 25, 51])

        # Set axis tick labels
        ax.set_xticklabels([1, 26, 52], fontsize=12, fontweight='bold')
        ax.set_yticklabels([1, 26, 52], fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Channels', fontsize=12, fontweight='bold')
        ax.set_ylabel('Channels', fontsize=12, fontweight='bold')
        # fig.colorbar(im, ax=ax)  # Add colorbar for each subplot

        plt.tight_layout()
        
        # Remove spines for effect size
        if idx>=num_view:
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)  # 设置边框的厚度为0.5

    # plt.savefig('/figures/connectivity.png')
    plt.savefig(output_fold+f'/statistics_responders_nonresponders_{name}.png')

    plt.show()            


time_points = ['pre_treatment', 'post_treatment', 'pre_minus_post_treatment']

pre_dmfc = np.load('allData/prognosis/DMFC/pre_treatment_hamd_reduction_50/data.npy')
pre_label = np.load('allData/prognosis/DMFC/pre_treatment_hamd_reduction_50/label.npy')

pre_post_dmfc = np.load('allData/prognosis/DMFC/pre_post_treatment_hamd_reduction_50/data.npy')
pre_post_label = np.load('allData/prognosis/DMFC/pre_post_treatment_hamd_reduction_50/label.npy')


for time in time_points:
    
    if time == 'pre_treatment':
        adj = pre_dmfc
        labels = pre_label
    elif time == 'post_treatment':
        adj = pre_post_dmfc[...,1]
        labels = pre_post_label
    elif time == 'pre_minus_post_treatment':
        adj = pre_post_dmfc[...,1] - pre_post_dmfc[..., 0]
        labels = pre_post_label
    
    show(adj, labels, time + f"subject_{labels.shape[0]}")