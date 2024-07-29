# Using default focal CrossEntropy to improve sensitivity, because I think the one I write can not focus y_true=0

# result :     'STL_depression_NCV_best_v4_Z1_channel-first' is the best 
# Using channel-last does not improve the performance
model_names=(
    'jamba_MTL'
)
config_names=(
    'STL_depression_NCV_best_v4y_nor-all'
    'STL_depression_NCV_best_v4y_nor-ind'
    'STL_depression_NCV_best_v4y_nor-01-magnitude-encoded'
    'STL_depression_NCV_best_v4y_nor-ind-magnitude-encoded'
    'STL_depression_NCV_best_v4y_nor-all-magnitude-encoded'
)
run_itr='jamba_20240725'
seeds=(31415926 27182818 16180339 12345678 98765432)
launcher_name="launcher.py"
db_file="results/experiment_results.db"






