#!/bin/bash
source scripts/experiment_configs/input_dimension_classification_depression.sh

# model='jamba_MTL'
# config_files=(
# 'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_512'
# 'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_128'
# )
# # 'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_16'
# # 'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_32'
# # 'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_64'
# # 'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_256'

# # 'MTL_all_hb_simple_all_1d_SPECIFY_FOLD_4_holdout_5_nor_loss'


# itr_name='MTL_20240710_VF3_AugmentRatio'
# seeds=(31415926 27182818 16180339 12345678 98765432)
# # python_file="./LOO_nested_CV_train_skf.py"
# python_file="./nested_CV_train.py"


# Loop through each seed
for seed in "${seeds[@]}"
do
    # Loop through each configuration file
    for config in "${config_files[@]}"
    do 
        # Construct the run_itr variable
        run_itr="${itr_name}_${seed}"

        # Construct the run_command
        run_command="conda run -n tf python $python_file $model $run_itr $config $seed"

        # Execute the run_command
        echo "Running command: $run_command"
        eval $run_command

        # Check if the command executed successfully
        if [ $? -ne 0 ]; then
            echo "Command failed: $run_command"
            exit 1
        fi

        sleep 1  # Adjust the sleep duration as needed    
    done
done

