#!/bin/bash

# Path to the SQLite database
db_file="/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/results/experiment_results.db"

# SQL command to delete all records from the experiments table
sqlite3 $db_file <<EOF
DELETE FROM experiments;
EOF

echo "All records have been deleted from the experiments table."
