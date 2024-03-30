# #!/bin/bash
# source ~/miniconda3/bin/activate tf
# sleep 2

cd /Users/shanxiafeng/Documents/Project/Research/fnirs-prognosis/code/fnirs-treatment-response-prediction
while true; do
    python ./LOO_nested_CV_train.py "$1" "$2" "$3"
    sleep 1  # Adjust the sleep duration as needed
done