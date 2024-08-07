import subprocess
import os 
import sys 

def check_have_enough_files(path, model_para):
    res_path = os.path.join(path, model_para)
    for loo in range(64):
        for k in range(5):
            flag_path = res_path + f'/LOO_nested_CV/LOO_{loo}/stratified_nested_5_CV_fold-{k}/test_acc.txt'
            if not os.path.exists(flag_path):
                print(f"Missing {flag_path}")
                return False
    return True

model = 'cnn_gnn_transformer' # gnn_transformer , comb_cnn(bad sensitivity), cnn_transformer(result is not consistent to gnn_transformer)
validation = 'loocv'
config = 'pretreatment_response_cv_5_mix_hb_'

msg = 'loocv_v3'

frontal_path = f'results/{model}/prognosis_mix_hb/pretreatment_response_frontal'
temporal_path = f'results/{model}/prognosis_mix_hb/pretreatment_response_temporal'

if not os.path.exists(frontal_path):
    os.makedirs(frontal_path)
if not os.path.exists(temporal_path):
    os.makedirs(temporal_path)
frontal_itr_count = os.listdir(frontal_path)
temporal_itr_count = os.listdir(temporal_path)
frontal_itr_count = [i for i in frontal_itr_count if i[:3] == 'loo']
temporal_itr_count = [i for i in temporal_itr_count if i[:3] == 'loo']


# if the itr amount of temoral is less than frontal, then run the temporal
if len(temporal_itr_count) <= len(frontal_itr_count):
    run_region = 'temporal'
    count = temporal_itr_count
    run_path = temporal_path
else:
    run_region = 'frontal'
    count = frontal_itr_count
    run_path = frontal_path
    
    
print(count)
# current itr is  
current_itr = len(count)-1
print(f"len(count): {current_itr}")

# if no loocv_v0 .... no need to check just run loocv_v0
if current_itr != -1:
    # check if the itr_{current_itr} have all LOO_63 and 5 fold
    for i in count:
        if not i[7:9].isdigit():
            itr = int(i[7])
        else:
            itr = int(i[7:9])
        if itr == current_itr:
            print(i)
            cuurent_itr_para = i
            
    res = check_have_enough_files(run_path, cuurent_itr_para)
else: 
    res = True 

if res: 
    run_itr = f"loocv_v{current_itr+1}"
else:
    run_itr = f"loocv_v{current_itr}"

config_file = config + run_region

run_command = f"conda run -n tf python ./LOO_nested_CV_train.py {model} {run_itr} {config_file}"#            bash_code = f"StratifiedKFold_holdout_train.py {model} automl"
subprocess.run(run_command, shell=True)

# nohup bash ./response_prediction.sh --model gnn_transformer --validation loocv --config pretreatment_response_cv_5_mix_hb_temporal --msg loocv_v3 > /dev/null 2>&1 &
# python ./LOO_nested_CV_train.py cnn_transformer loocv_v0 pretreatment_response_cv_5_mix_hb