
# nohup ./response_prediction.sh --model gnn_transformer --validation loocv --config posttreatment_response --msg test > /dev/null 2>&1 &
#!/bin/bash
# Activate conda environment
source ~/miniconda3/bin/activate tf
sleep 2

# Navigate to your project directory
# cd ~/Documents/fnirs/treatment_response/fnirs-depression-deeplearning

config_file="config.py"
# Process command line arguments
while [[ "$#" -gt 0 ]]; do
    echo $1
    case $1 in
        --validation) validation="$2"; shift ;;
        --model) model_name="$2"; shift ;;
        --msg) training_message="$2"; shift ;;
        --config) config_file="$2"; shift ;;
    esac
    shift
done
echo "Validation method: $validation"
echo "Model name: $model_name"
echo "Training message: $training_message"
echo "Config file: $config_file"

# Execute based on validation method
case $validation in
    loocv)
        while true; do
            python ./LOO_nested_CV_train.py "$model_name" "$training_message" "$config_file"
            sleep 1  # Adjust the sleep duration as needed
        done
        ;;
    holdout)
        while true; do
            python ./StratifiedKFold_holdout_train.py "$model_name" "$training_message"
            sleep 1  # Adjust the sleep duration as needed
        done
        ;;
    *)
        echo "Invalid or no validation method specified. Please specify --validation loocv or --validation holdout."
        exit 1
        ;;
esac
