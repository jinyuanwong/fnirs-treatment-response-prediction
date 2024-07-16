#!/bin/bash

# Function to log messages
# Function to log messages
log_message() {
    # Define the log file path
    log_file="./logs/experiment_log.txt"

    # Create the directory if it doesn't exist
    mkdir -p "$(dirname "$log_file")"

    # Append the message to the log file
    echo "$(date): $1" >> "$log_file"
}

# Function to check if a configuration file and seed combination has been executed
has_executed() {
    task_file=$1
    config_file=$2
    model=$3
    seed=$4
    result=$(sqlite3 $db_file "SELECT EXISTS(SELECT 1 FROM experiments WHERE task_file='$task_file' AND config_file='$config_file' AND seed=$seed AND model=$model AND status='completed');")
    echo $result
}

# Function to mark a configuration file and seed combination as executed
mark_executed() {
    task_file=$1
    config_file=$2
    model=$3
    seed=$4
    run_itr=$5
    sqlite3 $db_file "UPDATE experiments SET status='completed', end_time=CURRENT_TIMESTAMP WHERE task_file='$task_file' AND config_file='$config_file' AND seed=$seed AND model=$model AND run_itr='$run_itr';"
}

# Function to insert a new experiment record
insert_experiment() {
    task_file=$1
    config_file=$2
    model=$3
    seed=$4
    run_itr=$5
    sqlite3 $db_file "INSERT INTO experiments (task_file, config_file, seed, model, run_itr, status) VALUES ('$task_file', '$config_file', $seed, $model, '$run_itr', 'running');"
}
