# task: to investigate if the mlp can be very good (test AUC is 65.28 with a speed of 2.5 WO schedule)


model_names=(
    'jamba_MTL'
)
config_names=(
    'STL_depression_NCV_best_v2_WOschedule_only_mamba'
    'STL_depression_NCV_best_v2_WOschedule_only_mamba+1xattention'
    'STL_depression_NCV_best_v2_WOschedule_only_mlp'
    'STL_depression_NCV_best_v2_WOschedule_only_mlp+1xattention'    
    'STL_depression_NCV_best_v2_WOschedule_only_gnn'
    'STL_depression_NCV_best_v2_WOschedule_only_gnn+1xattention'    
    'STL_depression_NCV_best_v2_WOschedule_only_conv'
    'STL_depression_NCV_best_v2_WOschedule_only_conv+1xattention'
    'STL_depression_NCV_best_v2_WOschedule_only_mlp_conv1d+gnn+1xattention'
)
run_itr='jamba_20240722'
seeds=(31415926 27182818 16180339 12345678 98765432)
launcher_name="nested_CV_train.py"
db_file="results/experiment_results.db"


    # 'STL_depression_NCV_best_v2_WscheduleF01P8'
