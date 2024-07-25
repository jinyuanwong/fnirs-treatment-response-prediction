#!/bin/bash

# Path to the SQLite database
db_file="./results/experiment_results.db"

# Check if the database file exists
if [ ! -f "$db_file" ]; then
  echo "Error: Database file '$db_file' not found!"
  exit 1
fi

# SQL command to delete all records from all tables
sqlite3 $db_file <<EOF
DELETE FROM experiments;
DELETE FROM configs;
DELETE FROM models;
DELETE FROM performance;
DELETE FROM results;
EOF

# Check if the SQL command was successful
if [ $? -eq 0 ]; then
  echo "All records have been deleted from all tables."
else
  echo "Error: Failed to delete records from one or more tables."
  exit 1
fi