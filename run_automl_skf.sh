#!/bin/bash

# Define the variables
model='jamba'
config_file='mdd_classification_all_hb_simple_all_1d_SPECIFY_FOLD_4_holdout_5_nor' #'mdd_classification_all_hb_simple_all_1d_SPECIFY_FOLD_4_holdout_5_nor'
itr_name='aug_20_all_data_reduce_complex_1_2_only_gnn_find_random_seed' #'eval'

# Loop from 0 to 9
for itr in {1..2000}
do
    # Construct the run_itr variable
    run_itr="$itr_name$itr" #  $itr

    # Construct the run_command
    run_command="conda run -n tf python ./LOO_nested_CV_train_skf.py $model $run_itr $config_file"

    # Execute the run_command
    echo "Running command: $run_command"
    eval $run_command

    sleep 1  # Adjust the sleep duration as needed
done


# python ./LOO_nested_CV_train_skf.py jamba test_mlp_pre mdd_classification_jamba_light_data_mlp_preprocess_SKCV_4_holdout_5

# python ./LOO_nested_CV_train_skf.py jamba test mdd_classification_jamba_subject_110_hb_simple_all_1d_SPECIFY_FOLD_4_nor_raw

# python ./LOO_nested_CV_train_skf.py cnn_transformer retrain mdd_classification_mamba_subject_all_seq_ch_hb_simple_all_1d_SPECIFY_FOLD_3_holdout_4_nor

# python ./LOO_nested_CV_train_skf.py cnn_transformer eval_10 mdd_classification_all_hb_simple_all_1d_SPECIFY_FOLD_2_nor

# python ./LOO_nested_CV_train_skf.py jamba_hybrid eval_10 mdd_classification_jamba_hybrid_subject_all_hb_simple_all_1d_SPECIFY_FOLD_2_nor