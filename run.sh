#!/bin/bash

# Directory containing task files
# task_dir="scripts/experiment_tasks/test_sql.sh"

# Automatically populate task_files array with all .sh files in the task_dir
# task_files=($(find $task_dir -name '*.sh'))


    #     "tasks/model_states_depression.sh"
    #     "tasks/size_effect_of_dataset.sh"
    #     "tasks/input_dimension_classification_depression.sh"

task_files=(
    "tasks/20240716_replicate_result_and_compare_jamba_V1_V2_monitor_val_accuracy_enlarge_patience_function.sh"

)
# SQLite database to track executed configuration files
db_file="results/experiment_results.db"

# Source the utility functions
source utils/db_utils.sh

# Loop through each task file path
for task_file in "${task_files[@]}"
do
    # Source the current task file to get its configuration files
    source $task_file

    # Loop through each configuration file path defined in the task file
    for config_file in "${config_files[@]}"
    do
        log_message "Starting experiments with task file: $task_file, configuration file: $config_file"
        for model_name in "${models[@]}"
        do
            # Loop through each seed
            for seed in "${seeds[@]}"
            do
                # Check if the configuration file and seed combination has already been executed
                if [ $(has_executed "$task_file" "$config_file" "'$model_name'" "$seed") -eq 1 ]; then
                    echo "Skipping already executed task file, config file, and seed: $task_file, $config_file, $seed"
                    continue
                fi

                # Construct the run_itr variable
                run_itr="${itr_name}_${seed}"

                # Insert a new experiment record should be config instead of config_file
                insert_experiment "$task_file" "$config_file" "'$model_name'" "$seed" "$run_itr"

                # Construct the run_command
                run_command="conda run -n tf python $python_file '$model_name' $run_itr $config_file $seed"

                # Execute the run_command
                echo "Running command: $run_command"
                log_message "Running command: $run_command"
                eval $run_command

                # Check if the command executed successfully
                if [ $? -ne 0 ]; then
                    log_message "Command failed: $run_command"
                    echo "Command failed: $run_command"
                    sqlite3 $db_file "UPDATE experiments SET status='failed', end_time=CURRENT_TIMESTAMP WHERE task_file='$task_file' AND config_file='$config_file' AND seed=$seed AND model='$model_name' AND run_itr='$run_itr';"
                    exit 1
                fi
                # Mark the experiment as executed #  "$result"
                mark_executed "$task_file" "$config_file" "'$model_name'" "$seed" "$run_itr" 
                sleep 1  # Adjust the sleep duration as needed    
            done
        done
        log_message "Completed experiments with task file: $task_file, configuration file: $config_file"
    done
done

log_message "All experiments completed."