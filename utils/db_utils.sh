#!/bin/bash

generate_uuid() {
    uuid=$(uuidgen)
    echo $uuid
}

# Function to log messages
log_message() {
    # Define the log file path
    log_file="./logs/experiment_log.txt"

    # Create the directory if it doesn't exist
    mkdir -p "$(dirname "$log_file")"

    # Append the message to the log file
    echo "$(date): $1" >> "$log_file"
}

# Function to check or mark a configuration file and seed combination
execute_action() {
    local action=$1
    local task_name=$2
    local config_name=$3
    local model_name=$4
    local seed=$5
    local run_itr=$6
    local launcher_name=$7

    if [[ $action == "check" ]]; then
        result=$(sqlite3 $db_file "SELECT EXISTS(SELECT 1 FROM experiments WHERE task_name='$task_name' AND config_name='$config_name' AND seed=$seed AND model_name='$model_name' AND run_itr='$run_itr' AND launcher_name='$launcher_name' AND status='completed');")
        echo $result
    elif [[ $action == "mark" ]]; then
        sqlite3 $db_file "UPDATE experiments SET status='completed', end_time=CURRENT_TIMESTAMP WHERE task_name='$task_name' AND config_name='$config_name' AND seed=$seed AND model_name='$model_name' AND run_itr='$run_itr' AND launcher_name='$launcher_name';"
        echo "Marked as executed."
    else
        echo "Invalid action. Use 'check' or 'mark'."
    fi
}

# # Function to check if a configuration file and seed combination has been executed
# has_executed() {
#     task_name=$1
#     config_name=$2
#     model_name=$3
#     seed=$4
#     run_itr=$5
#     launcher_name=$6
#     result=$(sqlite3 $db_file "SELECT EXISTS(SELECT 1 FROM experiments WHERE task_name='$task_name' AND config_name='$config_name' AND seed=$seed AND model_name='$model_name' AND run_itr='$run_itr' AND launcher_name='$launcher_name' AND status='completed');")
#     echo $result
# }

# # Function to mark a configuration file and seed combination as executed
# mark_executed() {
#     task_name=$1
#     config_name=$2
#     model_name=$3
#     seed=$4
#     run_itr=$5
#     launcher_name=$6
#     sqlite3 $db_file "UPDATE experiments SET status='completed', end_time=CURRENT_TIMESTAMP WHERE task_name='$task_name' AND config_name='$config_name' AND seed=$seed AND model_name='$model_name' AND run_itr='$run_itr' AND launcher_name='$launcher_name';"
# }

# Generalized function to insert a record into the models table
# insert_record() {
#     local table_name=$1
#     local record_name=$2
#     local record_file_path=$3
#     local name_column="${table_name::-1}_name"
#     local file_column="${table_name::-1}_file"

#     # Check if the file exists
#     if [[ ! -f "$record_file_path" ]]; then
#         echo "Error: File '$record_file_path' not found."
#         return 1
#     fi

#     # Check if the record already exists
#     existing_record=$(sqlite3 $db_file "SELECT COUNT(*) FROM $table_name WHERE $name_column = '$record_name';")

#     if [[ $existing_record -eq 0 ]]; then
#         # Insert the new record
#         sqlite3 $db_file "INSERT INTO $table_name ($name_column, $file_column) VALUES ('$record_name', readfile('$record_file_path'));"
#         echo "Record '$record_name' inserted into the $table_name table."
#     else
#         echo "Record '$record_name' already exists in the $table_name table. Skipping insertion."
#     fi
# }

# Generalized function to insert or replace a record in the models table
insert_record() {
    local table_name=$1
    local record_name=$2
    local record_file_path=$3
    local name_column="${table_name::-1}_name"
    local file_column="${table_name::-1}_file"

    # Check if the file exists
    if [[ ! -f "$record_file_path" ]]; then
        echo "Error: File '$record_file_path' not found."
        return 1
    fi

    # Check if the record already exists
    existing_record=$(sqlite3 $db_file "SELECT COUNT(*) FROM $table_name WHERE $name_column = '$record_name';")

    if [[ $existing_record -eq 0 ]]; then
        # Insert the new record
        sqlite3 $db_file "INSERT INTO $table_name ($name_column, $file_column) VALUES ('$record_name', readfile('$record_file_path'));"
        echo "Record '$record_name' inserted into the $table_name table."
    else
        # Update the existing record
        sqlite3 $db_file "UPDATE $table_name SET $file_column = readfile('$record_file_path') WHERE $name_column = '$record_name';"
        echo "Record '$record_name' updated in the $table_name table."
    fi
}


insert_experiment() {
    task_name=$1
    config_name=$2
    model_name=$3
    seed=$4
    run_itr=$5
    launcher_name=$6
    experiment_id=$(generate_uuid)

    sqlite3 $db_file "INSERT INTO experiments (experiment_id, task_name, config_name, seed, model_name, run_itr, launcher_name, status) VALUES ('$experiment_id', '$task_name', '$config_name', $seed, '$model_name', '$run_itr', '$launcher_name', 'running');"
    echo $experiment_id
}

insert_performance() {
    seed=$1
    fold_name=$2
    history=$3
    y_val_true=$4
    y_val_pred=$5
    y_test_true=$6
    y_test_pred=$7
    checkpoint_path=${8:-NULL}
    others=${9:-NULL}

    performance_id=$(generate_uuid)

    sqlite3 $db_file "INSERT INTO performance (performance_id, seed, fold_name, history, y_val_true, y_val_pred, y_test_true, y_test_pred, checkpoint_path, others) VALUES ('$performance_id', $seed, '$fold_name', '$history', '$y_val_true', '$y_val_pred', '$y_test_true', '$y_test_pred', ${checkpoint_path:+'"'$checkpoint_path'"'}, ${others:+'"'$others'"'});"

    echo "Inserted performance with ID: $performance_id"
}

insert_result() {
    config_name=$1
    model_name=$2
    run_itr=$3
    python_file=$4
    performance_id=$5

    result_id=$(generate_uuid)

    sqlite3 $db_file "INSERT INTO results (result_id, config_name, model_name, run_itr, python_file, performance_id) VALUES ('$result_id', '$config_name', '$model_name', '$run_itr', '$python_file', '$performance_id');"

    echo "Inserted result with ID: $result_id"
}