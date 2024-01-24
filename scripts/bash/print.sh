#!/bin/bash

# Check if the script is called with two arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Please provide two arguments."
    exit 1
fi

# Save the arguments in variables
parameter=$1
findex=$2

# Check the value of the findex and execute the corresponding code block
case $findex in 
    1)
        for i in $(seq 0 9); do
            echo "Contents of df_metrics.csv in k-fold-$i:"
            cat Documents/JinyuanWang_pythonCode/results/$parameter/HbO-All-HC-MDD/k-fold-$i/df_metrics.csv
            echo
        done
        ;;
    2)
        for i in $(seq 0 9); do
            echo "Contents of df_metrics.csv in k-fold-$i:"
            cat Documents/JinyuanWang_pythonCode/results/$parameter/HbO-All-HC-MDD/k-fold-$i/best_hyperparameters.txt 
            echo
        done
        ;;
    *)
        echo "Invalid parameter for second index. Please use 1 or 2."
        exit 1
        ;;
esac
