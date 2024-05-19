import os
import pandas as pd
import matplotlib.pyplot as plt



def show_max_epoch_idxmax(
    main_path = 'results/gnn_transformer_with_cli_demo/prognosis_mix_hb/pretreatment_response/loocv_v0l1_rate_0.01_l2_rate_0.001_d_model_16_batch_size_64_n_layers_6/LOO_nested_CV/LOO_',
    TOTAL_SUBJECT = 64,
    NUM_OF_K_FOLD = 5 ,
    history_file_name = 'history.csv',
    best_model_based_metric = 'val_accuracy',
    inner_fold_name = 'stratified_nested_5_CV_fold-'
):
    max_epoch_arr = []
    idxmax_arr = []



    for sub in range(TOTAL_SUBJECT):
        for k in range(NUM_OF_K_FOLD):
            fold_path = main_path + str(sub) + '/' + inner_fold_name + str(k)
            file_path = fold_path + '/' + history_file_name
            if os.path.exists(file_path):
                res = pd.read_csv(file_path)
                max_index = len(res[best_model_based_metric]) # 
                indx_max = res[best_model_based_metric].idxmax()
                max_epoch_arr.append(max_index)
                idxmax_arr.append(indx_max)
                # see the performace of indx_max == 0 
                # if indx_max == 0:
                #     print(f"sub: {sub}, k: {k}, indx_max: {indx_max}")
                #     print(res.iloc[indx_max])
                
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hist(max_epoch_arr, bins=100)
    axs[0].set_title('max_epoch_arr')
    axs[1].hist(idxmax_arr, bins=100)
    axs[1].set_title('idxmax_arr')
    for i in range(2):
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel('Frequency')