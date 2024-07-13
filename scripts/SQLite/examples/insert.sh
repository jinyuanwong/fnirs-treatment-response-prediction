#!/bin/bash

# Path to the SQLite database
db_file="/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/results/experiment_results.db"

# Generate random values for the required fields
task_file="task_$(shuf -i 1-100 -n 1).sh"
config_file="config_$(shuf -i 1-100 -n 1).sh"
seed=$(shuf -i 1000-9999 -n 1)
run_itr="run_$(shuf -i 1-100 -n 1)"
result="result_$(shuf -i 1-100 -n 1)"
status="running"

# Insert the random record into the database
sqlite3 $db_file <<EOF
INSERT INTO experiments (task_file, config_file, seed, run_itr, result, status)
VALUES ('$task_file', '$config_file', $seed, '$run_itr', '$result', '$status');
EOF

echo "Inserted random record: task_file='$task_file', config_file='$config_file', seed=$seed, run_itr='$run_itr', result='$result', status='$status'"
