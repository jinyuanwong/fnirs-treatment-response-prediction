
## Experiment Task

### Description: this task is to investigate the effect of gcn layer in the model


model_names=(
    'jamba_MTL'
)
config_names=(
    'STL_depression_NCV_best_v2_only_gcn+conv+1xattention'
    'STL_depression_NCV_best_v2_only_gcn+1xattention'
    'STL_depression_NCV_best_v2_only_gcn'
)
run_itr='jamba_20240722'
seeds=(31415926 27182818 16180339 12345678 98765432)
launcher_name="nested_CV_train.py"
db_file="results/experiment_results.db"