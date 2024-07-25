# Using default focal CrossEntropy to improve sensitivity, because I think the one I write can not focus y_true=0

# result :     'STL_depression_NCV_best_v4_Z1_channel-first' is the best 
# Using channel-last does not improve the performance
model_names=(
    'jamba_MTL'
)
config_names=(
    'STL_depression_NCV_best_v4x_A1_ITR1-1e-5'
    'STL_depression_NCV_best_v4x_A1_ITR2-1e-5'
    'STL_depression_NCV_best_v4x_A1_ITR2-1e-6'
    'STL_depression_NCV_best_v4x_A1_ITR2-5e-6'
    'STL_depression_NCV_best_v4x_A1_ITR2-5e-7'
    'STL_depression_NCV_best_v4x_A2_ITR1-1e-5-best-val-acc-patiences30'
    'STL_depression_NCV_best_v4x_A2_ITR2-1e-7-best-val-acc-patiences30'
    'STL_depression_NCV_best_v4x_A2_ITR3-1e-4'
    'STL_depression_NCV_best_v4x_A2_ITR3-1e-6'
    'STL_depression_NCV_best_v4x_A2_ITR4-1e-5-fromITR3-1e-6'
    'STL_depression_NCV_best_v4x_A3_ITR1-best-val-loss'
    'STL_depression_NCV_best_v4x_A3_ITR2-best-val-loss'
    'STL_depression_NCV_best_v4x_A3_ITR2-best-val-accuracy'
)
run_itr='jamba_20240724_retrainModel'
seeds=(31415926 27182818 16180339 12345678 98765432)
launcher_name="nested_CV_train.py"
db_file="results/experiment_results.db"






