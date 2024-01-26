#!/bin/bash
source ~/miniconda3/bin/activate tf
sleep 2

cd ~/Documents/fnirs/treatment_response/fnirs-depression-deeplearning
# {0..9}
# while true; do
#     for i in 8 8 8 9 9 9; do
#         python ./diagnose.py 300 "$i" d fro
#         sleep 1  # Adjust the sleep duration as needed
#     done
# done

# while true; do
#     for i in 0 4; do
#         python ./train_transformer.py "$i" "$2"
#         sleep 1  # Adjust the sleep duration as needed
#     done
# done

# $1 -> model_name 
# $2 -> message for training
while true; do
    python ./LOOCV_train.py "$1" "$2"
    sleep 1  # Adjust the sleep duration as needed
done


