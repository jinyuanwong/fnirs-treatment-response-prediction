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






def show_hb_type(data, label, hb_type_name, figname):
    task_start_index = 100 
    task_end_index = 700

    responders = data[label==1]
    nonresponders = data[label==0]

    all_p = np.zeros((3,52))
    all_effect_size = np.zeros((3,52))
    colors = ['red','white']

    cmap = LinearSegmentedColormap.from_list('RedToWhite', colors, N=256)
    title = ['Task activation of '+hb_type_name, 'Mean of '+hb_type_name, 'Task change of '+hb_type_name]
    plt.figure(figsize=(25,15))
    plt.subplot(2, 1, 1)  # 1 row, 2 columns, select the 1st subplot
    plt.title('HCs vs MDDs (P-value)', fontsize=20, fontweight='bold')
    for channel in range(52):
        channel_p = []
        channel_effect_size = []

        responders_task_change = np.mean(responders[:,task_end_index:, channel], axis=1) - np.mean(responders[:,0:task_start_index, channel], axis=1)
        nonresponders_task_change = np.mean(nonresponders[:,task_end_index:, channel], axis=1) - np.mean(nonresponders[:,0:task_start_index, channel], axis=1)

        responders_mean_hbo = np.mean(responders[..., channel], axis=1)
        nonresponders_mean_hbo = np.mean(nonresponders[..., channel], axis=1)
        responders_task_rest_hbo = np.mean(responders[:, task_start_index:task_end_index, channel], axis=1) - np.mean(responders[:,0:task_start_index, channel], axis=1) - np.sum(responders[:,task_end_index:-1, channel], axis=1)
        nonresponders_task_rest_hbo = np.mean(nonresponders[:,task_start_index:task_end_index, channel], axis=1) - np.mean(nonresponders[:,0:task_start_index, channel], axis=1) - np.sum(nonresponders[:, task_end_index:-1, channel], axis=1)
        # print(f'test: {np.mean(nonresponders[:,0:task_start_index, channel], axis=1) - np.mean(nonresponders[:, task_end_index:-1, channel], axis=1)}')
        # print(nonresponders_task_rest_hbo.shape)
        
        # Create a figure and axis
        # fig, ax = plt.subplots(1, 4, figsize=(16,8))
        for index, (responders_hbo, nonresponders_hbo) in enumerate([(responders_task_change, nonresponders_task_change), (responders_mean_hbo, nonresponders_mean_hbo), (responders_task_rest_hbo, nonresponders_task_rest_hbo)]):
            data = [responders_hbo, nonresponders_hbo]

            stat, p_value = mannwhitneyu(responders_hbo,nonresponders_hbo)
            channel_p.append(p_value)
            

            g1 = pg.compute_effsize(responders_hbo,nonresponders_hbo, eftype='Hedges')
            channel_effect_size.append(g1)
        
        # add the FDR here 
        all_p[:,channel] = channel_p
        all_effect_size[:,channel] = channel_effect_size
            
        #     print(f'{p_value} | ', end='')
        # print()
        
        # 使用指数型的p_value颜色变化

    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.4)
    



    # Create the figure and axes
    # plt.figure(figsize=(25,15))

    # Define the height at which the markers will be placed (above the heatmap)
    marker_height = all_p.shape[0]  # You can adjust this as needed


    for indices, color in zip([PSFC_indices, DPC_indices, STG_indices, VPC_indices, MPC_indices], [PSFC_color, DPC_color, STG_color, VPC_color, MPC_color]):
        for index in indices:
            # Fill a small rectangle (vertical span) above each channel index
            # Adjust the 'width' to control the span width of each colored area
            width = 1  # Adjust this width to your preference
            plt.axvspan(index, index+1, ymin=0.8, ymax=1, color=color, zorder=1)


    # plt.axvspan(1, 2, ymin=0.85, ymax=1, color='red', zorder=1)

    # print(marker_height/(marker_height+1))
    # # Draw the color markers first
    # plt.fill_between(x=np.arange(25, 53), y1=marker_height+0.25, y2=marker_height+1, color='purple', label='Rear Brain')

    # 在 marker_height 位置添加一条黑线
    plt.axhline(y=marker_height+0.05, color='black', linewidth=0.4)
    plt.axvline(x=1, ymin=0, ymax=0.8, color='black', linewidth=0.4)
    plt.axvline(x=53, ymin=0, ymax=0.8, color='black', linewidth=0.4)
    # plt.axvline(x=53, ymin=0, ymax=0.8, color='black', linewidth=0.4)


    # Adjust the y-axis limits to accommodate the markers
    plt.ylim([0, marker_height+1])

    fdr_p = np.zeros(all_p.shape)
    for i in range(all_p.shape[0]):
        tmp = all_p[i]
        _, adjusted_all_p, _, _ = multipletests(tmp, alpha=0.05, method='fdr_bh')
        fdr_p[i] = adjusted_all_p
    all_p = fdr_p
    # Now, plot the heatmap on top of the color markers
    im = plt.imshow(all_p, norm=LogNorm(vmin=0.001, vmax=0.05), cmap=cmap, extent=[1,53, 0, marker_height])

    ax = plt.gca()
    # print(f'all_p: {all_p}')
    cbar = plt.colorbar(im, ax=ax, fraction=0.01, pad=0.02, ticks=[0.001, 0.01, 0.05])
    cbar.ax.set_yticklabels(['0.001', '0.01', '0.05'], fontsize=12, fontweight='bold')

    # cbar = plt.colorbar(im, ax=ax, fraction=0.01, pad=0.02, ticks=[0.00001, 0.0001, 0.001, 0.01, 0.05])
    # cbar.ax.set_yticklabels(['0.00001', '0.0001', '0.001', '0.01', '0.05'], fontsize=12, fontweight='bold')
    # 获取颜色条的位置信息
    cbar_pos = cbar.ax.get_position()

    # 修改位置信息以将高度减少一半
    new_height = cbar_pos.height / 4
    new_y = cbar_pos.y0 + new_height*1.5  # 更新y位置以使颜色条保持居中

    cbar.ax.set_position([cbar_pos.x0, new_y, cbar_pos.width, new_height])
    # cbar.ax.set_yticklabels(['0.001', '0.01', '0.05'], fontsize=12, fontweight='bold')
    #  设置颜色条的刻度标签为科学计数法格式
    # cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

    # 定义一个函数，用于格式化刻度标签
    def fmt(x, pos):
        if x == 0:  # 避免对0取对数
            return '0'
        if x == 0.05:
            return r'$5 \times 10^{-2}$'
        return r'$10^{%d}$' % np.round(np.log10(x))

    # 使用FuncFormatter
    # cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))

    # Set the ticks and labels as required
    plt.xticks([ i + 0.5 for i in [1, 26, 52]])
    plt.gca().set_xticklabels([1, 26, 52], fontsize=15, fontweight='bold')  # Correcting labels to display 1 to 52
    plt.yticks([0.5,1.5,2.5], title, fontsize=16, fontweight='bold')

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




    plt.xlim([1, 53.01])
    # Show the plot
    # plt.show()
    
    plt.subplot(2, 1, 2)  # 1 row, 2 columns, select the 1st subplot
    plt.title('HCs vs MDDs (Effect size)', fontsize=20, fontweight='bold')

    for indices, color in zip([PSFC_indices, DPC_indices, STG_indices, VPC_indices, MPC_indices], [PSFC_color, DPC_color, STG_color, VPC_color, MPC_color]):
        for index in indices:
            # Fill a small rectangle (vertical span) above each channel index
            # Adjust the 'width' to control the span width of each colored area
            width = 1  # Adjust this width to your preference
            plt.axvspan(index, index+1, ymin=0.8, ymax=1, color=color, zorder=1)


    # plt.axvspan(1, 2, ymin=0.85, ymax=1, color='red', zorder=1)

    print(marker_height/(marker_height+1))
    # # Draw the color markers first
    # plt.fill_between(x=np.arange(25, 53), y1=marker_height+0.25, y2=marker_height+1, color='purple', label='Rear Brain')

    # 在 marker_height 位置添加一条黑线
    plt.axhline(y=marker_height+0.05, color='black', linewidth=0.4)
    plt.axvline(x=1, ymin=0, ymax=0.8, color='black', linewidth=0.4)
    plt.axvline(x=53, ymin=0, ymax=0.8, color='black', linewidth=0.4)
    # plt.axvline(x=53, ymin=0, ymax=0.8, color='black', linewidth=0.4)


    # Adjust the y-axis limits to accommodate the markers
    plt.ylim([0, marker_height+1])


    colors = ['#448196', 'white', '#c45c3d']

    # Create the custom color map
    cmap_effect_size = LinearSegmentedColormap.from_list('CustomColorMap', colors, N=256)
    im = plt.imshow(all_effect_size, vmin=-0.5, vmax=0.5, cmap=cmap_effect_size, extent=[1,53, 0, marker_height])

    ax = plt.gca()

    cbar = plt.colorbar(im, ax=ax, fraction=0.01, pad=0.02, ticks=[-0.5, -0.25, 0, 0.25, 0.5])
    cbar.ax.set_yticklabels(['-0.5', '-0.25', '0', '0.25', '0.5'], fontsize=12, fontweight='bold')
    # 获取颜色条的位置信息
    cbar_pos = cbar.ax.get_position()

    # 修改位置信息以将高度减少一半
    new_height = cbar_pos.height / 4
    new_y = cbar_pos.y0 + new_height*1.5  # 更新y位置以使颜色条保持居中

    cbar.ax.set_position([cbar_pos.x0, new_y, cbar_pos.width, new_height])


    # Set the ticks and labels as required
    plt.xticks([ i + 0.5 for i in [1, 26, 52]])
    plt.gca().set_xticklabels([1, 26, 52], fontsize=15, fontweight='bold')  # Correcting labels to display 1 to 52
    plt.yticks([0.5,1.5,2.5], title, fontsize=16, fontweight='bold')

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




    plt.xlim([1, 53.01])
    # Show the plot
    # plt.savefig(output_fold+f'/statistics_responders_nonresponders_{figname}.png')
    plt.show()

DATA = np.load('allData/diagnosis/hb_data.npy')
LABEL = np.load('allData/diagnosis/label.npy')

pre_post_data_combine_type = 'substract'

output_fold = 'FigureTable/statics/timedomain'
import os
if not os.path.exists(output_fold):
    os.makedirs(output_fold)
# name_of_input = ['pre_treatment', 'post_treatment', 'pre_minus_post_treatment']
name_of_input = ['HCs vs MDDs']
for fig_name in name_of_input:
    print(fig_name)
    if fig_name =='HCs vs MDDs':
        print('entering - 1')
        data = DATA
        label = LABEL

    print('data.shape', data.shape)
    HbO = np.transpose(data[...,0],(0,2,1))
    HbR = np.transpose(data[...,1],(0,2,1))
    HbT = np.transpose(data[...,2],(0,2,1))
    # For pre - treatment HAMD reduction 50 - HbO 
    show_hb_type(HbO, label, 'HbO', fig_name + '_HbO' + '_subject' + str(data.shape[0]))
    # For pre - treatment HAMD reduction 50 - HbR
    show_hb_type(HbR, label, 'HbR', fig_name + '_HbR' + '_subject' + str(data.shape[0]))
    # For pre - treatment HAMD reduction 50 - HbT 
    show_hb_type(HbT, label, 'HbT', fig_name + '_HbT' + '_subject' + str(data.shape[0]))

    
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

