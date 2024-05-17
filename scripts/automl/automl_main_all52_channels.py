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

model = 'gnn_transformer_with_cli_demo' # gnn_transformer , comb_cnn(bad sensitivity), cnn_transformer(result is not consistent to gnn_transformer), cnn_gnn_transformer
validation = 'loocv'
config = 'pretreatment_response_cv_5_mix_hb'
dataset = 'prognosis_mix_hb'
task = 'pretreatment_response'

all_region_path = f'results/{model}/{dataset}/{task}'

if not os.path.exists(all_region_path):
    os.makedirs(all_region_path)

all_region_itr_count = os.listdir(all_region_path)
print('all_region_itr_count', all_region_itr_count)
all_region_itr_count = [i for i in all_region_itr_count if i[:3] == 'loo']

# if the itr amount of temoral is less than frontal, then run the temporal

run_region = 'all_region'
count = all_region_itr_count
run_path = all_region_path
    
    
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
config_file = config

run_command = f"conda run -n tf python ./LOO_nested_CV_train.py {model} {run_itr} {config_file}"#            bash_code = f"StratifiedKFold_holdout_train.py {model} automl"
subprocess.run(run_command, shell=True)

# nohup bash ./response_prediction.sh --model gnn_transformer --validation loocv --config pretreatment_response_cv_5_mix_hb_temporal --msg loocv_v3 > /dev/null 2>&1 &
# python ./LOO_nested_CV_train.py cnn_transformer loocv_v0 pretreatment_response_cv_5_mix_hb
# 