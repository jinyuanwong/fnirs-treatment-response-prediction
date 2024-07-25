# 



models=(
'jamba_MTL'
)
config_names=(
'NCV_STL_depression_best_all_checkpoint_monitor_val_loss_begin_patience_15_dropout_rate_0_3'
'NCV_STL_depression_best_all_checkpoint_monitor_val_accuracy_begin_patience_15_dropout_rate_0_3'
)
itr_name='jamba_20240717'
seeds=(31415926 27182818 16180339 12345678 98765432)
# python_file="./LOO_nested_CV_train_skf.py"
python_file="./nested_CV_train.py"