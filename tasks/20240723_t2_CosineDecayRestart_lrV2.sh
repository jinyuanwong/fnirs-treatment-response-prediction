# Using default focal CrossEntropy to improve sensitivity, because I think the one I write can not focus y_true=0



model_names=(
    'jamba_MTL'
)
config_names=(
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-1step10'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-1step50'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-1step100'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-2step10'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-2step50'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-2step100'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-3step10'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-3step50'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-3step100'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-4step10'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-4step50'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-4step100'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-5step10'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-5step50'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-5step100'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-6step10'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-6step50'
    'STL_depression_NCV_best_v4_Y1_MLP_lr1e-6step100'
)
run_itr='jamba_20240723'
seeds=(31415926 27182818 16180339 12345678 98765432)
launcher_name="nested_CV_train.py"
db_file="results/experiment_results.db"





