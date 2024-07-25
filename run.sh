#!/bin/bash

# Directory containing task files
# task_dir="scripts/experiment_tasks/test_sql.sh"

# Automatically populate task_files array with all .sh files in the task_dir
# task_files=($(find $task_dir -name '*.sh'))


    #     "tasks/model_states_depression.sh"
    #     "tasks/size_effect_of_dataset.sh"
    #     "tasks/input_dimension_classification_depression.sh"

task_names=(
    "20240725_update_framwork"
)
# SQLite database to track executed configuration files
db_file="results/experiment_results.db"

# Source the utility functions
source utils/db_utils.sh

# Loop through each task file path
for task_name in "${task_names[@]}"
do
    # Source the current task file to get its configuration files
    task_name_path="tasks/$task_name.sh"
    source $task_name_path
    insert_record "tasks" "$task_name" "$task_name_path"
    
    launcher_name_path="./$launcher_name"
    insert_record "launchers" "$launcher_name" "$launcher_name_path"

    # Loop through each configuration file path defined in the task file
    for config_name in "${config_names[@]}"
    do
        config_name_path="configs/$config_name.py"
        insert_record "configs" "$config_name" "$config_name_path"

        log_message "Starting experiments with task file: $task_name, configuration file: $config_name"
        for model_name in "${model_names[@]}"
        do
                        
            model_name_path="classifiers/$model_name.py" 
            insert_record "models" "$model_name" "$model_name_path"

            # Loop through each seed
            for seed in "${seeds[@]}"
            do
                # Check if the configuration file and seed combination has already been executed
                # should consider running situation!!
                if [ $(execute_action "check" "$task_name" "$config_name" "$model_name" "$seed" "$run_itr" "$launcher_name") -eq 1 ]; then
                    echo "Skipping already executed task file, config file, and seed: $task_name, $config_name, $seed"
                    continue
                fi

                # Construct the run_itr variable
                # run_itr="${itr_name}_${seed}"

                # Insert a new experiment record should be config instead of config_name
                experiment_id=$(insert_experiment "$task_name" "$config_name" "$model_name" "$seed" "$run_itr" "$launcher_name")

                # Construct the run_command
                run_command="conda run -n tf python $launcher_name_path $model_name $run_itr $config_name $seed $experiment_id"

                # Execute the run_command
                echo "Running command: $run_command"
                log_message "Running command: $run_command"
                eval $run_command

                # Check if the command executed successfully
                if [ $? -ne 0 ]; then
                    log_message "Command failed: $run_command"
                    echo "Command failed: $run_command"
                    sqlite3 $db_file "UPDATE experiments SET status='failed', end_time=CURRENT_TIMESTAMP WHERE task_name='$task_name' AND config_name='$config_name' AND seed=$seed AND model_name='$model_name' AND run_itr='$run_itr' AND launcher_name='$launcher_name';"
                    exit 1
                fi
                # Mark the experiment as executed #  "$result"
                execute_action "mark" $task_name $config_name $model_name $seed $run_itr $launcher_name
                sleep 1  # Adjust the sleep duration as needed    
            done
        done
        log_message "Completed experiments with task file: $task_name, configuration file: $config_name"
    done


done

log_message "All experiments completed."



