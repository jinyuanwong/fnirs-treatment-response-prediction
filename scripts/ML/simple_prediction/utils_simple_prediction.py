import numpy as np 
import os
import sys 
import pandas as pd
import json

def set_path():
    if sys.platform == 'darwin':
        print("Current system is macOS")
        main_fold_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction'
    elif sys.platform == 'linux':
        print("Current system is Ubuntu")
        main_fold_path = '/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning'
    else:
        print("Current system is neither macOS nor Ubuntu")
    os.chdir(main_fold_path)
    
def load_data_for_classification():
    data_pth = 'allData/prognosis_mix_hb/pretreatment_response/nor_demo_cli_merge_data.npy'
    label_pth = 'allData/prognosis_mix_hb/pretreatment_response/label.npy'
    data = np.load(data_pth)
    labels = np.load(label_pth)
    return data, labels

def load_data_for_MDD_HC_classification(modality):
    
    dataset_name = 'diagnosis514'
    
    if modality == 'hbo':
        data_pth = f'allData/{dataset_name}/nine_regions_hbo_task_change_fnirs_features.npy'
    elif modality == 'hbr':
        data_pth = f'allData/{dataset_name}/nine_regions_hbr_task_change_fnirs_features.npy'    
    else: 
        data_pth = f'allData/{dataset_name}/nine_regions_hbt_task_change_fnirs_features.npy'
        
    label_pth = 'allData/{dataset_name}/label.npy'
    
    data = np.load(data_pth)
    labels = np.load(label_pth)
    
    save_fold = 'results/ML_results/simple_prediction/MDD_HC_classification/{dataset_name}/' + f'nine_regions_{modality}_task_change_fnirs_features/'
    
    if not os.path.exists(save_fold):
        os.makedirs(save_fold)
    
    return data, labels, save_fold

def load_data_for_MDD_HC_classification_extracted_features(modality):
    
    dataset_name = 'diagnosis514'#'diagnosis514' #diagnosis_110_fabeha_dataset_hb_all
    
    
    data_pth = f'allData/{dataset_name}/sig_normalized_feature.npy'
        
    label_pth = f'allData/{dataset_name}/label.npy'
    
    data = np.load(data_pth)
    labels = np.load(label_pth)
    
    save_fold = f'results/ML_results/simple_prediction/MDD_HC_classification/{dataset_name}/' + f'extracted_feature_{modality}/'
    
    if not os.path.exists(save_fold):
        os.makedirs(save_fold)
    
    return data, labels, save_fold

def load_data_for_partial_response_prediction():
    label_pth = 'allData/prognosis_mix_hb/pretreatment_response/label_partial_response.npy'
    labels = np.load(label_pth)
    return labels


def add_cgi(data):
    cgi_path = 'allData/prognosis_mix_hb/pretreatment_response/nor_T2_SDS_CGI.npy'
    cgi = np.load(cgi_path)
    data = np.concatenate((data, cgi[:, :6]), axis=1)
    return data

def add_mddr(data):
    mddr_path = 'allData/prognosis_mix_hb/pretreatment_response/MDDR/MDDR_derived_from_load_evaluate.npy'
    mddr = np.load(mddr_path)
    mddr = mddr[..., -1]
    mddr = np.transpose(mddr, (1, 0))
    mddr = mddr.mean(axis=1)[..., np.newaxis]
    data = np.concatenate((data, mddr), axis=1)
    return data


def load_task_change_data(region_name='nine_regions'):
    data_pth = 'allData/prognosis_mix_hb/pretreatment_response/task_change_hb_{}.npy'.format(region_name)
    data = np.load(data_pth)
    return data

def load_mean_fnirs_data(region_name='nine_regions'):
    data_pth = 'allData/prognosis_mix_hb/pretreatment_response/mean_hb_{}.npy'.format(region_name)
    data = np.load(data_pth)
    return data



def load_fnirs_feature_data(region_name = 'nine_regions'):
    if region_name == 'nine_regions':
        # data = task_change_nine_regions = load_task_change_data()[..., 2]
        data = mean_nine_regions = load_mean_fnirs_data(region_name)[..., 2]
        # data = np.concatenate((task_change_nine_regions, mean_nine_regions), axis=1)
    if region_name == 'five_regions':
        task_change_five_regions = load_task_change_data(region_name)[..., 2]
        mean_five_regions = load_mean_fnirs_data(region_name)[..., 2]
        data = np.concatenate((task_change_five_regions, mean_five_regions), axis=1)
    return data

def add_task_change_data(data, index=2):
    """
    index: 0 - HbO, 1 - HbR, 2 - HbT
    """
    task_change_data = load_task_change_data()[..., index]
    data = np.concatenate((data, task_change_data), axis=1)
    return data

def load_data():
    data_pth = 'allData/prognosis_mix_hb/pretreatment_response/merge_psychiatry_demo_dose_data.npy'
    label_pth = 'allData/prognosis_mix_hb/pretreatment_response/label_hamd.npy'
    data = np.load(data_pth)
    labels = np.load(label_pth)
    labels = change_hamd_to_percent(labels)
    return data, labels

def save_into_csv(csv_filename, arr):
    headers = ["random_seed", "validation_acc", "validation_sen", "validation_spe", "validation_auc",
           "test_acc", "test_sen", "test_spe", "test_auc"]
    df = pd.DataFrame([arr], columns=headers)
    file_exists = os.path.isfile(csv_filename)
    if not file_exists:
        # If file does not exist, write data with headers   
        df.to_csv(csv_filename, index=False, mode='w')
        print(f"Data saved to {csv_filename} with headers.")
    else:
        # If file exists, append data without headers
        df.to_csv(csv_filename, index=False, mode='a', header=False)
        print(f"Data appended to {csv_filename} without headers.")
        

# Function to save results
def save_results_pred(model_name, seed, y_pred, y_true, shap_values, filename):
    # Load existing results if the file exists
    try:
        with open(filename, 'r') as file:
            results = json.load(file)
    except FileNotFoundError:
        results = {}

    # Create an entry for the current model and seed
    key = f"{model_name}_seed_{seed}"
    results[key] = {
        'y_pred': y_pred.tolist(),  # Convert to list for JSON serialization
        'y_true': y_true.tolist(),  # Convert to list for JSON serialization
        'shap': shap_values.tolist()
    }

    # Save the updated results back to the file
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)

        
def save_model_seed(save_fold, random_seed, save_result):

    model_name = [i for i in save_result['inner_result']]
    for model in model_name:
        val_result = [round(val, 4) for key, val in save_result['inner_result'][model].items()]
        test_result = [round(val, 4) for key, val in save_result['external_result'][model].items() if key[-4:] != 'test']
        val_test_arr = [random_seed] +  val_result + test_result
        save_path = save_fold + model + '.csv'
        
        y_pred = save_result['external_result'][model]['y_pred_test']
        y_true = save_result['external_result'][model]['y_true_test']
        if save_result['external_result'][model].get('shap_values_test') is not None:
            shap_values = save_result['external_result'][model]['shap_values_test']
        else:
            shap_values = np.array([])
        save_results_pred(model, random_seed, y_pred, y_true, shap_values, save_fold + model + '_pred.json')
        
        save_into_csv(save_path, val_test_arr)
