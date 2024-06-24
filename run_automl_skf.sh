#!/bin/bash

# Define the variables
model='jamba_multimodal'
config_file='mdd_classification_jamba'

# Loop from 0 to 9
for itr in {1..9}
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
