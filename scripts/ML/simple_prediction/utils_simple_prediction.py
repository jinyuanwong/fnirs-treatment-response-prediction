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
    data = np.concatenate((data, mddr), axis=1)
    return data


def load_task_change_data():
    data_pth = 'allData/prognosis_mix_hb/pretreatment_response/task_change_hb.npy'
    data = np.load(data_pth)
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
def save_results_pred(model_name, seed, y_pred, y_pred_prob, filename):
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
        'y_pred_prob': y_pred_prob.tolist()  # Convert to list for JSON serialization
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
        y_pred_prob = save_result['external_result'][model]['y_true_test']

        save_results_pred(model, random_seed, y_pred, y_pred_prob, save_fold + model + '_pred.json')
        
        save_into_csv(save_path, val_test_arr)
