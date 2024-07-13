#!/bin/bash

# Path to the SQLite database
db_file="/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/results/experiment_results.db"

# Query to select all records from the experiments table
sqlite3 $db_file <<EOF
.headers on
.mode column
SELECT * FROM experiments;
EOF
