# Using default focal CrossEntropy to improve sensitivity, because I think the one I write can not focus y_true=0



model_names=(
    'jamba_MTL'
)
config_names=(
    'STL_depression_NCV_best_v3_WOschedule_only_convTgnnT1xattention-CE' 
    'STL_depression_NCV_best_v3_WOschedule_only_convTgnnT1xattention-FL' 
    'STL_depression_NCV_best_v3_WOschedule_only_convTmlpTgnn-CE' 
    'STL_depression_NCV_best_v3_WOschedule_only_convTmlpTgnn-FL' 
    'STL_depression_NCV_best_v3_WOschedule_only_convTmlpTgnnT1xattention-CE' 
    'STL_depression_NCV_best_v3_WOschedule_only_convTmlpTgnnT1xattention-FL' 
    'STL_depression_NCV_best_v3_WOschedule_only_mlp-CE' 
    'STL_depression_NCV_best_v3_WOschedule_only_mlp-FL' 
    'STL_depression_NCV_best_v3_WOschedule_only_mlpT1xattention-FL' 
    'STL_depression_NCV_best_v3_WOschedule_only_mlpT1xattention-CE'
)
run_itr='jamba_20240722'
seeds=(31415926 27182818 16180339 12345678 98765432)
launcher_name="nested_CV_train.py"
db_file="results/experiment_results.db"



# configs/STL_depression_NCV_best_v4_Y1_MLP_lr1e-1.py configs/STL_depression_NCV_best_v4_Y1_MLP_lr1e-2.py configs/STL_depression_NCV_best_v4_Y1_MLP_lr1e-3.py configs/STL_depression_NCV_best_v4_Y1_MLP_lr1e-4.py configs/STL_depression_NCV_best_v4_Y1_MLP_lr1e-5.py configs/STL_depression_NCV_best_v4_Y1_MLP_lr1e-6.py