
model_names=(
    'jamba_MTL'
)
config_names=(
    'STL_depression_NCV_best_v1_wo_mamba_w_mlp_w_conv_norm_batchnorm'
    'STL_depression_NCV_best_v1_wo_mamba_w_mlp_w_conv_norm_rmsnorm'
    'STL_depression_NCV_best_v1_wo_mamba_w_mlp_w_conv_norm_layernorm'
)
run_itr='jamba_20240720'
seeds=(31415926 27182818 16180339 12345678 98765432)
launcher_name="nested_CV_train.py"
db_file="results/experiment_results.db"
