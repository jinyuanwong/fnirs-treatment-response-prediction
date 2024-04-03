#!/bin/bash
# source ~/miniconda3/bin/activate tf
# sleep 2

cd ~/Documents/JinyuanWang_pythonCode

while true; do
    python ./train_transformer.py "$1" "$2"
    sleep 1  # Adjust the sleep duration as needed
done


