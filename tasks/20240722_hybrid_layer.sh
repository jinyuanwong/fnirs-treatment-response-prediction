
model_names=(
    'jamba_MTL'
)
config_names=(
    'STL_depression_NCV_best_v2_only_conv1d+1xattention'
    'STL_depression_NCV_best_v2_only_conv1d+gnn+1xattention'
    'STL_depression_NCV_best_v2_only_gnn+1xattention'
)
run_itr='jamba_20240722'
seeds=(31415926 27182818 16180339 12345678 98765432)
launcher_name="nested_CV_train.py"
db_file="results/experiment_results.db"
