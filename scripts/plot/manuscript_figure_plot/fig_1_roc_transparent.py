import numpy as np 
import os 
import sys 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

if sys.platform == 'darwin':
    print("Current system is macOS")
    main_fold_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction'
elif sys.platform == 'linux':
    print("Current system is Ubuntu")
    main_fold_path = '/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning'
else:
    print("Current system is neither macOS nor Ubuntu")
os.chdir(main_fold_path)

interval = 2
def plot_avg_auc(fprs, tprs, roc_aucs, title):
    mean_fpr = np.linspace(0, 1, 100)[::interval]

    # Interpolate TPRs at these common FPR levels
    mean_tpr = np.zeros_like(mean_fpr)
    tpr_interpolated = []

    for i in range(len(fprs)):
        tpr_interp = np.interp(mean_fpr, fprs[i], tprs[i])
        tpr_interpolated.append(tpr_interp)
        
    # Calculate the mean TPR
    tpr_interpolated = np.array(tpr_interpolated)
    mean_tpr = tpr_interpolated.mean(axis=0)
    std_tpr = tpr_interpolated.std(axis=0)

    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    # Compute AUC
    mean_auc = np.mean(roc_aucs)
    std_auc = np.std(roc_aucs)
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, lw=4, color='darkviolet')  # Dark purple color
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='plum', alpha=0.3) 
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.grid()

    # plt.title('AUROC Curve')

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.savefig('FigureTable/manuscript_figures/roc_curve{}.png'.format(title), transparent=True)
    plt.show()

# Generate random data for 10 repetitions
# 

for i in range(4):
    np.random.seed(42 + i)
    n_reps = 10
    fprs = []
    tprs = []
    roc_aucs = []
    for _ in range(n_reps):
        fpr = np.linspace(0, 1, 100)
        tpr = 1. * np.sin(fpr * np.pi / 2) + 0.2 * fpr**2 + np.random.normal(0, 0.02, 100)*5
        fpr = fpr[::interval]
        tpr = tpr[::interval]
        tpr = np.clip(tpr, 0, 1)  # Ensure TPR is within [0, 1]
        roc_auc = np.trapz(tpr, fpr)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)

    # Plot the average AUC
    plot_avg_auc(fprs, tprs, roc_aucs, i)