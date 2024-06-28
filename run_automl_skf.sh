#!/bin/bash

# Define the variables
model='jamba'
config_file='mdd_classification_jamba_subject_110_hb_simple_all_1d_SPECIFY_FOLD_5_nor_chkpt'

# Loop from 0 to 9
for itr in {1..5}
do
    # Construct the run_itr variable
    run_itr="eval$itr"

    # Construct the run_command
    run_command="conda run -n tf python ./LOO_nested_CV_train_skf.py $model $run_itr $config_file"

    # Execute the run_command
    echo "Running command: $run_command"
    eval $run_command

    sleep 1  # Adjust the sleep duration as needed
done
