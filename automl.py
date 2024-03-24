import subprocess
import re
import os 
import pandas as pd


def change_para_base_on_file(file_path, para_name, new_value):
    para_name = para_name + '='
    new_value = str(new_value)
    # Define the path to the file
    

    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Replace the value
    with open(file_path, 'w') as file:
        for line in lines:
            if para_name in line:
                # Splitting the line at 'lr=' and replacing the existing value with the new value
                prefix, old_value = line.split(para_name)
                print(prefix, old_value)
                new_line = prefix + para_name + new_value + '\n'
                file.write(new_line)
            else:
                file.write(line)
                
def start_to_train(bash_code):
    subprocess.run(bash_code, shell=True)
    
kfold_nums = [3, 4, 5, 6, 7, 8, 9, 10]
models = ['chao_cfnn', 'li_svm', 'yu_gnn', 'zhu_xgboost', 'wang_alex', 'gnn_transformer']

config_path = 'config.py'

while True:
    for model in models:
        if model == 'gnn_transformer':
            input_hb_type = ['prognosis/pre_treatment_hamd_reduction_50'] 
        else:
            input_hb_type = ['prognosis/pretreatment_benchmarks']
        for k_fold in kfold_nums:
            change_para_base_on_file(config_path, 'INPUT_HB_TYPE', input_hb_type)
            change_para_base_on_file(config_path, 'SPECIFY_FOLD', k_fold)
            bash_code = f"conda run -n tf python StratifiedKFold_holdout_train.py {model} automl"
            start_to_train(bash_code)