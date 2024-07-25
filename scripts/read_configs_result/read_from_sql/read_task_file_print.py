import numpy as np 
import sys
import os
import re
import pandas as pd 
import sqlite3



def set_path():
    if sys.platform == 'darwin':
        print("Current system is macOS")
        main_fold_path = '/Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction'
    elif sys.platform == 'linux':
        print("Current system is Ubuntu")
        main_fold_path = '/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning'
        # main_fold_path = '/root/autodl-tmp/fnirs-treatment-response-prediction'
    else:
        print("Current system is neither macOS nor Ubuntu")
    sys.path.append(main_fold_path)
    os.chdir(main_fold_path)
    
def connect_to_database_and_fetch_results(database_path, query, params):
    # Connect to the database
    conn = sqlite3.connect(database_path)
    
    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()
    
    # Execute the SQL query
    cursor.execute(query, params)
    
    # Fetch all rows from the result set
    rows = cursor.fetchall()
    
    # Get the column names
    column_names = [description[0] for description in cursor.description]
    
    # Close the cursor and the database connection
    cursor.close()
    conn.close()
    
    # Convert rows to list of dictionaries
    results = []
    for row in rows:
        results.append(dict(zip(column_names, row)))
    
    # Return the list of dictionaries representing each row
    return results    
    
    
def read_table_from_database(database_path, table_name, table_id=None):
    # Connect to the database
    conn = sqlite3.connect(database_path)
    
    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()
    
    # Construct the SQL query
    if table_id is None:
        # Fetch all rows if no ID is specified
        cursor.execute(f"SELECT * FROM {table_name}")
    else:
        # Fetch specific row based on ID
        cursor.execute(f"SELECT * FROM {table_name} WHERE {table_name[:-1]}_id=?", (table_id,))
    
    # Fetch all rows from the result set
    rows = cursor.fetchall()
    
    # Get the column names
    column_names = [description[0] for description in cursor.description]
    
    # Close the cursor and the database connection
    cursor.close()
    conn.close()
    
    # Convert rows to list of dictionaries
    results = []
    for row in rows:
        results.append(dict(zip(column_names, row)))
    
    # Return the list of dictionaries representing each row
    return results




def read_result_from_sql(result_id, DATABASE_PATH):
    """ 
    
    
    return val_metrics, test_metrics
    
    val_metrics: list: (5) [AUC, Accuracy, Sensitivity, Specificity, Duration]
    test_metrics: list: (5) [AUC, Accuracy, Sensitivity, Specificity, Duration]

    """
    result_val_metrics = []
    result_test_metrics = []
    result = read_table_from_database(DATABASE_PATH, "results", result_id)[0] # should only have one record for each result_id
    # for result in  read_table_from_database(DATABASE_PATH, "results", result_id):
        # fold_test_metrics = []
        # fold_val_metrics = []
    
    num_of_fold_per_seed = len(result['performance_ids'].split(','))
    if num_of_fold_per_seed != 20:
        raise ValueError(f"Number of folds per seed should be 20 (outer-5, inner-4), but got {num_of_fold_per_seed}")
    for performance_id in result['performance_ids'].split(','):
        performance = read_table_from_database(DATABASE_PATH, "performances", performance_id)[0]
        history = pd.read_json(performance['history'])
        
        # Append val_metrics DataFrame to the list
        result_val_metrics.append(pd.read_json(performance['val_performance_metrics']))
        result_test_metrics.append(pd.read_json(performance['test_performance_metrics']))

    conc_result_val_metrics = pd.concat(result_val_metrics, axis=0)
    conc_result_test_metrics = pd.concat(result_test_metrics, axis=0)
    avg_result_val_metrics = conc_result_val_metrics.mean(axis=0)
    avg_result_test_metrics = conc_result_test_metrics.mean(axis=0)
    
    val_metrics = [
                   avg_result_val_metrics['accuracy'], 
                   avg_result_val_metrics['sensitivity'], 
                   avg_result_val_metrics['specificity'], 
                   avg_result_val_metrics['AUC'], 
                   avg_result_val_metrics['duration']]
    test_metrics = [
                   avg_result_test_metrics['accuracy'], 
                   avg_result_test_metrics['sensitivity'], 
                   avg_result_test_metrics['specificity'], 
                   avg_result_test_metrics['AUC'], 
                   avg_result_test_metrics['duration']]    
    # print("Mean of result val_metrics")
    # print(val_metrics)
    # print("Mean of result test_metrics")
    # print(test_metrics)
    return val_metrics, test_metrics
    
def read_config_from_experiment(experiment_args, DATABASE_PATH):
    # Create the SQL query with placeholders for parameters
    args_all = [f"{key} = ?" for key in experiment_args.keys()]
    str_args_all = ' AND '.join(args_all)
    sql_query = f"SELECT * FROM experiments WHERE {str_args_all}"
    params = tuple(experiment_args.values())    
    config_experiment = connect_to_database_and_fetch_results(DATABASE_PATH, sql_query, params)
    if len(config_experiment) == 0:
        raise ValueError(f"No experiment found with the following parameters: {experiment_args}")
    elif len(config_experiment) < 5:
        raise ValueError(f"Number of experiments found is {len(config_experiment)}, should be 5")
    # print('config_experiment', len(config_experiment))
    config_result = { 
        'val': [],
        'test': []
        }
    # Loop Seed
        # each config_experiment should have num_of_seed experiment(result)
        
    for experiment in config_experiment:
        if experiment['result_id'] is None: continue
        val_metric, test_metric = read_result_from_sql(experiment['result_id'], DATABASE_PATH)
        
        config_result['val'].append(val_metric)
        
        config_result['test'].append(test_metric)
        # print(f"config: {experiment['result_id']} test_metric: {test_metric[0]}")
    avg_val = np.mean(config_result['val'], axis=0)
    avg_test = np.mean(config_result['test'], axis=0)
    return avg_val, avg_test


def extract_task_file_content(task_file_path):
    """ 
    Args:
    task_file_path: str: 'tasks/20240720_normalization_method.sh'
    
    
    returns (experiment_args, config_names)
    
    experiment_args: dict: {'model_name': 'jamba_MTL', 
                            'run_itr': 'jamba_20240720', 
                            'launcher_name': 'nested_CV_train.py'}
                            
    configs_name: list of str: ['STL_depression_NCV_best_v1_wo_mamba_w_mlp_w_conv_norm_batchnorm',
                                'STL_depression_NCV_best_v1_wo_mamba_w_mlp_w_conv_norm_rmsnorm',
                                'STL_depression_NCV_best_v1_wo_mamba_w_mlp_w_conv_norm_layernorm']
    """
    experiment_args = {}
    config_names = []

    with open(task_file_path, 'r') as file:
        lines = file.readlines()
    in_model_names = False
    in_config_names = False

    for line in lines:
        line = line.strip()
        print('line', line)
        if line.startswith('model_names'):
            in_model_names = True
            continue

        if in_model_names:
            if line.startswith(')'):
                in_model_names = False
                continue
            model_name = line.strip().strip("'")
            experiment_args['model_name'] = model_name
            in_model_names = False
            continue

        if line.startswith('config_names'):
            in_config_names = True
            continue

        if in_config_names:
            if line.startswith(')'):
                in_config_names = False
                continue
            config_name = line.strip().strip("'")
            if config_name:
                config_names.append(config_name)
            continue

        if line.startswith('run_itr'):
            run_itr = line.split('=')[1].strip().strip("'")
            experiment_args['run_itr'] = run_itr

        if line.startswith('launcher_name'):
            launcher_name = line.split('=')[1].strip().strip('"')
            experiment_args['launcher_name'] = launcher_name

    # Derive task_name from the file name
    experiment_args['task_name'] = task_file_path.split('/')[-1].split('.')[0]
    experiment_args['status'] = 'completed'
    return experiment_args, config_names
    
def read_task_file(task_file_path):    
# task_file_path = 'tasks/20240720_normalization_method.sh'    
    experiment_args, config_names = extract_task_file_content(task_file_path)

    # print('experiment_args', experiment_args)
    # print('config_names', config_names)


    PARAMETER_NAME = experiment_args['task_name']
    plot_evaluation_metrics_header(table_name = 'Depression', parameter_name=PARAMETER_NAME, val_auc_threshold=0)     

    # Loop Config
    for config_name in config_names:
        # print('config_name', config_name)
        experiment_args['config_name'] = config_name
        
        val_metrics, test_metrics = read_config_from_experiment(experiment_args, DATABASE_PATH)
        print_md_table_val_test_AUC(config_name.split('_')[-1], val_metrics, test_metrics, print_table_header=False, already_balanced_accuracy=False)

# set_path() 
# from utils.utils_mine import plot_evaluation_metrics_header
# from utils.fnirs_utils import print_md_table_val_test_AUC
# DATABASE_PATH = "results/experiment_results.db"
# task_file_path = "tasks/20240721_model_seed_optimization.sh"
# read_task_file(task_file_path)
if __name__ == '__main__':
    set_path() 
    from utils.utils_mine import plot_evaluation_metrics_header
    from utils.fnirs_utils import print_md_table_val_test_AUC
    DATABASE_PATH = "results/experiment_results.db"
    task_file_path = sys.argv[1] if len(sys.argv) > 1 else "tasks/20240724_t2_retrain_multiple_time.sh"
    if os.path.exists(task_file_path):
        read_task_file(task_file_path)
    else:
        raise ValueError(f"Task file {task_file_path} does not exist")
