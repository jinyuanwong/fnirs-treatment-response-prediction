#!/bin/bash

# Path to the SQLite database
db_file="./results/experiment_results.db"

# Query to select all records from the experiments table
sqlite3 $db_file <<EOF
.headers on
.mode column

SELECT * FROM experiments;




EOF
# SELECT * FROM results;
# SELECT * FROM models;

# SELECT * FROM tasks;
# SELECT * FROM configs;
# SELECT * FROM models;
# SELECT * FROM experiments;
# SELECT * FROM performances;
# SELECT * FROM launchers;
# 