"""
This script computes the statistical significance of responder and non-responder
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

from matplotlib.colors import LogNorm
from scipy.stats import mannwhitneyu,spearmanr
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.multitest import multipletests
import matplotlib.ticker as ticker
PSFC_location = ['C9', 'C10', 'C20', 'C21', 'C1', 'C2', 'C11', 'C12']
PSFC_color = "#00ff37"
# Dorsolateral prefrontal cortex
DPC_location = ['C7','C8', 'C17', 'C18', 'C19', 'C28', 'C29', 'C3', 'C4', 'C13', 'C14', 'C15', 'C24', 'C25']
DPC_color = "#62fff3"
#Superior temporal gyrus
STG_location = ['C22', 'C23', 'C32', 'C33', 'C43', 'C44', 'C30', 'C31', 'C41', 'C42', 'C51', 'C52']
STG_color = "#d27ced"
# Ventrolateral prefrontal cortex
VPC_location = ['C34', 'C35', 'C45', 'C46', 'C39', 'C40', 'C49', 'C50']
VPC_color = "#ff992c"
# Medial prefrontal cortex
MPC_location = ['C5', 'C6', 'C16', 'C26', 'C27', 'C36', 'C37', 'C38', 'C47', 'C48']  
MPC_color = "#ffd682"
# 首先，我们定义一个函数来将通道位置转换为索引
def location_to_index(locations):
    return [int(loc[1:]) for loc in locations]

# 将位置转换为索引
PSFC_indices = location_to_index(PSFC_location)
DPC_indices = location_to_index(DPC_location)
STG_indices = location_to_index(STG_location)
VPC_indices = location_to_index(VPC_location)
MPC_indices = location_to_index(MPC_location)


def get_nine_region_data(data):
    def get_channel_index_of_region(ch_name):
        return np.array([int(ch_name[1:])-1 for ch_name in ch_name])

    # Posterior superior frontal cortex
    # PSFC_ch = ['C9', 'C10', 'C20', 'C21', 'C1', 'C2', 'C11', 'C12'] # 
    left_PSFC_location = ['C9', 'C10', 'C20', 'C21']
    right_PSFC_location = ['C1', 'C2', 'C11', 'C12']

    # Dorsolateral prefrontal cortex
    # DPC_ch = ['C7','C8', 'C17', 'C18', 'C19', 'C28', 'C29', 'C3', 'C4', 'C13', 'C14', 'C15', 'C24', 'C25']
    left_DPC_location = ['C7','C8', 'C17', 'C18', 'C19', 'C28', 'C29']
    right_DPC_location = ['C3', 'C4', 'C13', 'C14', 'C15', 'C24', 'C25']

    #Superior temporal gyrus
    # STG_ch = ['C22', 'C23', 'C32', 'C33', 'C43', 'C44', 'C30', 'C31', 'C41', 'C42', 'C51', 'Cnum_of_region'] #
    left_STG_location = ['C22', 'C23', 'C32', 'C33', 'C43', 'C44']
    right_STG_location = ['C30', 'C31', 'C41', 'C42', 'C51', 'C52']

    # Ventrolateral prefrontal cortex
    # VPC_ch = ['C34', 'C35', 'C45', 'C46','C39', 'C40', 'C49', 'C50'] # 
    left_VPC_location = ['C34', 'C35', 'C45', 'C46']
    right_VPC_location = ['C39', 'C40', 'C49', 'C50']

    # Medial prefrontal cortex
    MPC_location = ['C5', 'C6', 'C16', 'C26', 'C27', 'C36', 'C37', 'C38', 'C47', 'C48']  
    

    all_region_location = [left_PSFC_location, right_PSFC_location, left_DPC_location, right_DPC_location, left_STG_location, right_STG_location, left_VPC_location, right_VPC_location, MPC_location]
    all_region_location = [get_channel_index_of_region(i) for i in all_region_location]
    print(len(all_region_location))
    print(all_region_location)


    nine_region_data = np.zeros((data.shape[0], len(all_region_location), data.shape[2]))

    for i, region_ch in enumerate(all_region_location):
        region_data = data[:,region_ch,:]
        region_data = np.mean(region_data, axis=1)
        nine_region_data[:,i,:] = region_data
    return nine_region_data

def modify_char_position(cbar):
    
    cbar_pos = cbar.ax.get_position()

    # 修改位置信息以将高度减少一半
    new_height = cbar_pos.height  / 7 * 5
    new_y = cbar_pos.y0 + cbar_pos.height/7 #+ new_height  # 更新y位置以使颜色条保持居中

    cbar.ax.set_position([cbar_pos.x0, new_y, cbar_pos.width, new_height])

def compute_p_value_effect_size(data, label, task_start_index, task_end_index):
    data_shape = data.shape
    responders = data[label==1]
    nonresponders = data[label==0]
    all_p = np.zeros((3,data_shape[2]))
    all_effect_size = np.zeros((3,data_shape[2]))
    
    for channel in range(data_shape[2]):
        channel_p = []
        channel_effect_size = []
        responders_task_change = np.mean(responders[:,task_end_index:, channel], axis=1) - np.mean(responders[:,0:task_start_index, channel], axis=1)
        nonresponders_task_change = np.mean(nonresponders[:,task_end_index:, channel], axis=1) - np.mean(nonresponders[:,0:task_start_index, channel], axis=1)

        responders_mean_hbo = np.mean(responders[..., channel], axis=1)
        nonresponders_mean_hbo = np.mean(nonresponders[..., channel], axis=1)
        
        responders_task_rest_hbo = np.mean(responders[:, task_start_index:task_end_index, channel], axis=1) - np.mean(responders[:,0:task_start_index, channel], axis=1) - np.mean(responders[:,task_end_index:-1, channel], axis=1)
        nonresponders_task_rest_hbo = np.mean(nonresponders[:,task_start_index:task_end_index, channel], axis=1) - np.mean(nonresponders[:,0:task_start_index, channel], axis=1) - np.mean(nonresponders[:, task_end_index:-1, channel], axis=1)
        # print(f'test: {np.mean(nonresponders[:,0:task_start_index, channel], axis=1) - np.mean(nonresponders[:, task_end_index:-1, channel], axis=1)}')
        # print(nonresponders_task_rest_hbo.shape)
        
        # Create a figure and axis
        # fig, ax = plt.subplots(1, 4, figsize=(16,8))
        for index, (responders_hbo, nonresponders_hbo) in enumerate([(responders_task_change, nonresponders_task_change), (responders_mean_hbo, nonresponders_mean_hbo), (responders_task_rest_hbo, nonresponders_task_rest_hbo)]):
            data = [responders_hbo, nonresponders_hbo]

            stat, p_value = mannwhitneyu(responders_hbo,nonresponders_hbo)
            channel_p.append(p_value)
            

            g1 = pg.compute_effsize(nonresponders_hbo, responders_hbo, eftype='Hedges')
            channel_effect_size.append(g1)
        
        # add the FDR here 
        all_p[:,channel] = channel_p
        all_effect_size[:,channel] = channel_effect_size
        
    return all_p, all_effect_size

def plot_pretreatment_analysis(ax, data, data_shape, hb_type_name):
    plt.title('Nonresponders vs. Responders (Effect size)', fontsize=20, fontweight='bold')
    y_label_name = ['Task activation of '+hb_type_name, 'Mean of '+hb_type_name, 'Task change of '+hb_type_name]

    for indices, color in zip(np.arange(1, 1+data_shape[2]), [PSFC_color, PSFC_color, DPC_color, DPC_color, STG_color, STG_color, VPC_color, VPC_color, MPC_color]):
        plt.axvspan(indices, indices+1, ymin=0.8, ymax=1, color=color, zorder=1)

    print(marker_height/(marker_height+1))
    # # Draw the color markers first
    # plt.fill_between(x=np.arange(25, 53), y1=marker_height+0.25, y2=marker_height+1, color='purple', label='Rear Brain')

    # 在 marker_height 位置添加一条黑线
    plt.axhline(y=marker_height+0.05, color='black', linewidth=0.4)
    plt.axvline(x=1, ymin=0, ymax=0.8, color='black', linewidth=0.4)
    plt.axvline(x=data_shape[2]+1, ymin=0, ymax=0.8, color='black', linewidth=0.4)
    # plt.axvline(x=53, ymin=0, ymax=0.8, color='black', linewidth=0.4)


    # Adjust the y-axis limits to accommodate the markers
    plt.ylim([0, marker_height+1])


    colors = ['#448196', 'white', '#c45c3d']

    # Create the custom color map
    cmap_effect_size = LinearSegmentedColormap.from_list('CustomColorMap', colors, N=256)
    im = plt.imshow(data, vmin=-0.5, vmax=0.5, cmap=cmap_effect_size, extent=[1,data_shape[2]+1, 0, marker_height])

    ax = plt.gca()

    cbar = plt.colorbar(im, ax=ax, fraction=0.01, pad=0.02, ticks=[-0.5, -0.25, 0, 0.25, 0.5])
    cbar.ax.set_yticklabels(['-0.5', '-0.25', '0', '0.25', '0.5'], fontsize=12, fontweight='bold')
    modify_char_position(cbar)

    # Set the ticks and labels as required
    plt.xticks([ i + 0.5 for i in [1, data_shape[2]//2, data_shape[2]]])
    plt.gca().set_xticklabels([1, data_shape[2]//2, data_shape[2]], fontsize=15, fontweight='bold')  # Correcting labels to display 1 to 52
    plt.yticks([0.5,1.5,2.5], y_label_name, fontsize=16, fontweight='bold')

    # 隐藏上边框
    plt.gca().spines['top'].set_visible(False)

    plt.gca().spines['left'].set_visible(False)

    plt.gca().spines['right'].set_visible(False)


    # 为其他三个边框设置线宽，颜色
    for spine_position, spine in plt.gca().spines.items():
        if spine_position in ['left', 'right', 'bottom']:
            spine.set_linewidth(0.4)
            spine.set_color('black')
            
            
    # Add colorbar and other plot settings as required
    # plt.colorbar(shrink=0.2)

    plt.xlim([1, 0.01+data_shape[2]])
    
def show_hb_type(data, label, hb_type_name, fig_name, task_start_index, task_end_index, using_fdr):
    data_shape = data.shape
    
    p_value, effect_size= compute_p_value_effect_size(data, label, task_start_index, task_end_index)
    
    print('p_value: ', p_value) 
    print('effect_size: ', effect_size)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    plot_pretreatment_analysis(axes[0], p_value, data_shape, hb_type_name)
    plot_pretreatment_analysis(axes[1], effect_size, data_shape, hb_type_name)
    plt.show()
    

    

    
DATA =  np.load('allData/prognosis_mix_hb/pretreatment_response/hb_data.npy')
LABEL =  np.load('allData/prognosis_mix_hb/pretreatment_response/label.npy')

pre_post_data_combine_type = 'substract'

output_fold = 'FigureTable/statics/timedomain/prognosis/hitachi_process'
# output_fold = 'FigureTable/statics/timedomain/numOfSubject_140_zhifei_process'

import os
if not os.path.exists(output_fold):
    os.makedirs(output_fold)
# name_of_input = ['pre_treatment', 'post_treatment', 'pre_minus_post_treatment']
name_of_input = ['Nonresponders vs. Responders']
nine_region_name = ['L-PSFC', 'R-PSFC', 'L-DPC', 'R-DPC', 'L-STG', 'R-STG', 'L-VPC', 'R-VPC', 'MPC']
for fig_name in name_of_input:
    print(fig_name)
    if fig_name =='Nonresponders vs. Responders':
        print('entering - 1')
        data = DATA
        label = LABEL



# data[subject1] - mean(data[subject1])
    def individual_normalization(data):
        for i in range(data.shape[0]):
            data[i] = (data[i] - np.mean(data[i])) / np.std(data[i])
        return data
    # data = individual_normalization(data)
    
    data = get_nine_region_data(data)
    
    HbO = np.transpose(data[...,0::2],(0,2,1))
    # HbO = individual_normalization(HbO)
    print(f'HbO: {HbO.shape}')
    HbR = np.transpose(data[...,1::2],(0,2,1))
    # HbR = individual_normalization(HbR)
    HbT = HbO + HbR
    

    plt.figure()     
    nonresponder_hbo = HbO[label==0]
    responder_hbo = HbO[label==1]    
    nonresponder_hbr = HbR[label==0]
    responder_hbr = HbR[label==1]
    plt.subplot(1,2,1)
    plt.plot(np.mean(nonresponder_hbo, axis=(0,2)), label=f"Nonresponders {nonresponder_hbo.shape[0]}")
    plt.plot(np.mean(responder_hbo, axis=(0,2)), label=f"Responders {responder_hbo.shape[0]}")
    
    plt.subplot(1,2,2)
    plt.plot(np.mean(nonresponder_hbr, axis=(0,2)), label=f"Nonresponders {nonresponder_hbr.shape[0]}")
    plt.plot(np.mean(responder_hbr, axis=(0,2)), label=f"Responders {responder_hbr.shape[0]}")
    
    plt.legend()
    plt.title('Average HbO and HbR of Responders and Nonresponders')
    plt.savefig(output_fold+'/Nonresponders vs. Responders.png')
    
    for using_fdr in [True, False]:
       
        # For pre - treatment HAMD reduction 50 - HbO 
        show_hb_type(HbO, label, 'HbO', fig_name + '_HbO' + '_subject' + str(data.shape[0]), 100, 700, using_fdr)
        # For pre - treatment HAMD reduction 50 - HbR
        show_hb_type(HbR, label, 'HbR', fig_name + '_HbR' + '_subject' + str(data.shape[0]), 100, 700, using_fdr)
        # For pre - treatment HAMD reduction 50 - HbT 
        show_hb_type(HbT, label, 'HbT', fig_name + '_HbT' + '_subject' + str(data.shape[0]), 100, 700, using_fdr)

    
# for data, label in zip([pre_data, pre_post_data], [pre_label, pre_post_label]):
    
#     print(data.shape)
#     if data.shape[-2] == 2:
#         # which is pre_post data 
#         if pre_post_data_combine_type == 'substract':
#             data = data[..., 1] - data[..., 0]

#     HbO = data[...,0]
#     HbR = data[...,1]
#     HbT = HbO + HbR
#     print(HbO.shape)
#     # For pre - treatment HAMD reduction 50 - HbO 
#     show_hb_type(HbO, label, 'HbO')
#     # For pre - treatment HAMD reduction 50 - HbR
#     show_hb_type(HbR, label, 'HbR')
#     # For pre - treatment HAMD reduction 50 - HbT 
#     show_hb_type(HbT, label, 'HbT')

