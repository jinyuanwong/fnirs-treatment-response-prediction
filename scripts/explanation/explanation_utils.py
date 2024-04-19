import numpy as np 
import os 
import sys 

import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter
from matplotlib import cm

def gradientbars(bars,ydata,cmap):
    ax = bars[0].axes
    lim = ax.get_xlim()+ax.get_ylim()
    ax.axis(lim)
    for bar in bars:
        bar.set_facecolor("none")
        x,y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        grad = np.atleast_2d(np.linspace(0,1*h/max(ydata),256)).T
        ax.imshow(grad, extent=[x,x+w,y,y+h], origin='lower', aspect="auto", 
                  norm=cm.colors.NoNorm(vmin=0,vmax=1), cmap=plt.get_cmap(cmap))

"""

input will be something like [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
return colors ['red', 'green', ...]

"""
def get_fnirs_channel_colors(ranked_channel):
    colors = []
    # asign color to channel 
    dpc_color = 'lightblue'
    stg_color = 'purple'
    vpc_color = (1.0, 0.5, 0.0)
    mpc_color = (1.0, 0.64, 0.0)
    psfc_color = 'green'

    # Posterior superior frontal cortex
    PSFC_ch = ['C9', 'C10', 'C20', 'C21', 'C1', 'C2', 'C11', 'C12'] # left_PSFC_location = ['C9', 'C10', 'C20', 'C21'] , right_PSFC_location = ['C1', 'C2', 'C11', 'C12']

    # Dorsolateral prefrontal cortex
    DPC_ch = ['C7','C8', 'C17', 'C18', 'C19', 'C28', 'C29', 'C3', 'C4', 'C13', 'C14', 'C15', 'C24', 'C25']# left_DPC_location = ['C7','C8', 'C17', 'C18', 'C19', 'C28', 'C29'], right_DPC_location = ['C3', 'C4', 'C13', 'C14', 'C15', 'C24', 'C25']

    #Superior temporal gyrus
    STG_ch = ['C22', 'C23', 'C32', 'C33', 'C43', 'C44', 'C30', 'C31', 'C41', 'C42', 'C51', 'C52'] #left_STG_location = ['C22', 'C23', 'C32', 'C33', 'C43', 'C44'], right_STG_location = ['C30', 'C31', 'C41', 'C42', 'C51', 'C52']

    # Ventrolateral prefrontal cortex
    VPC_ch = ['C34', 'C35', 'C45', 'C46','C39', 'C40', 'C49', 'C50'] # left_VPC_location = ['C34', 'C35', 'C45', 'C46'], right_VPC_location = ['C39', 'C40', 'C49', 'C50']

    # Medial prefrontal cortex
    MPC_location = ['C5', 'C6', 'C16', 'C26', 'C27', 'C36', 'C37', 'C38', 'C47', 'C48']  


    for ch in ranked_channel:
        ch_name = 'C' + str(ch+1) 
        if ch_name in PSFC_ch:
            colors.append(psfc_color)
        elif ch_name in DPC_ch:
            colors.append(dpc_color)
        elif ch_name in STG_ch:
            colors.append(stg_color)
        elif ch_name in VPC_ch:
            colors.append(vpc_color)
        elif ch_name in MPC_location:
            colors.append(mpc_color)
        else:
            print('ch not found', ch_name)
            colors.append('grey')
            
    return colors

def show_ranked_shap_channel_importance(data, y_label_name='mean |SHAP value|'):
    
    # Calculate both mean and standard deviation along the specified axes
    channel_importance_mean = np.mean(data, axis=(0))
    channel_importance_std = np.std(data, axis=(0))

    # Ranking the channels by their importance (mean values)
    ranked_channel = np.argsort(channel_importance_mean) 
    ranked_channel = ranked_channel[::-1]
    print("Ranked channels by importance:", ranked_channel)

    # 
    ranked_channel_x = ['C' + str(i+1) for i in ranked_channel]
    ranked_channel_importance = channel_importance_mean[ranked_channel]
    ranked_channel_importance_std = channel_importance_std[ranked_channel]
    error = [np.zeros(channel_importance_std.shape), ranked_channel_importance_std]  # First row zeros, second row stds

    fig, ax = plt.subplots(figsize=(20,10))

    shap_bar = ax.bar(ranked_channel_x, ranked_channel_importance, yerr=error, capsize=5, edgecolor='grey')



    yticks = [0, 0.5, 1, 1.2]
    yticklabels = ['0', '0.5', '1', ' ']
    # yticks = [0, 1e-4, 2e-4, 3e-4]
    # yticklabels = ['0', '0.0001', '0.0002', '0.0003']
    plt.yticks(yticks, yticklabels,fontsize=20)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.get_offset_text().set_fontsize(15)

    # plt.ylim(0, 0.00025)  # Optional: Set y-axis limits if necessary

    plt.xticks(np.arange(52), fontsize=15, rotation=45, fontweight='bold')  # Optional: Improve x-axis readability if necessary
    plt.ylabel(y_label_name,fontsize=15, fontweight='bold')
    plt.xlabel('Ranked channel',fontsize=15, fontweight='bold')
    plt.title('Average Channel Importance', fontweight='bold', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels

    gradientbars(shap_bar, ranked_channel_importance, 'viridis_r')


    # marker the region color of each channel
    colors = get_fnirs_channel_colors(ranked_channel)
    y_marker_position = 1.2 * 0.95
    for bar, color in zip(shap_bar, colors):
        # Calculate the center of the bar
        center_x = bar.get_x() + bar.get_width() / 2
        
        # Use scatter to add a colored square marker
        ax.scatter(center_x, y_marker_position, s=400, color=color, marker='s', zorder=3)

    plt.show()