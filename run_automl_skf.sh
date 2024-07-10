#!/bin/bash

model='jamba_MTL'
config_files=(
'MTL_all_hb_simple_all_1d_SPECIFY_FOLD_4_holdout_5_nor_STL_gender_AUG_1'
)
# 'MTL_all_hb_simple_all_1d_SPECIFY_FOLD_4_holdout_5_nor_loss'


itr_name='MTL_20240710_V2_AugmentRatio_1'
seeds=(31415926 27182818 16180339 12345678 98765432)
# python_file="./LOO_nested_CV_train_skf.py"
python_file="./nested_CV_train.py"
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

# #!/bin/bash

# model='jamba_MTL'
# config_files=(
# 'MTL_all_hb_simple_all_1d_SPECIFY_FOLD_4_holdout_5_nor_loss'
# 'MTL_all_hb_simple_all_1d_SPECIFY_FOLD_4_holdout_5_nor'
# )

# itr_name='MTL_V1_240708'
# seeds=(1720351610 1720353641 1720355734 1720358054 1720360139)
# python_file="./LOO_nested_CV_train_skf.py"

# # Get the length of the arrays 
# num_seeds=${#seeds[@]}
# num_configs=${#config_files[@]}


# # Loop through indices
# for seed_idx in $(seq 0 $((num_seeds-1)))
# do
#     for config_idx in $(seq 0 $((num_configs-1)))
#     do 
#         seed="${seeds[$seed_idx]}"
#         config="${config_files[$config_idx]}"
        
#         # Construct the run_itr variable
#         run_itr="${itr_name}_${seed}" # Use indices

#         # Construct the run_command
#         run_command="conda run -n tf python $python_file $model $run_itr $config $seed"

#         # Execute the run_command
#         echo "Running command: $run_command"
#         eval $run_command

#         sleep 1  # Adjust the sleep duration as needed    
#     done
# done


# Loop from 0 to 9
# for itr in {1..5}
# do
#     # Construct the run_itr variable
#     run_itr="$itr_name$itr" #  $itr $itr

#     # Construct the run_command
#     run_command="conda run -n tf python ./LOO_nested_CV_train_skf.py $model $run_itr $config_file"

#     # Execute the run_command
#     echo "Running command: $run_command"
#     eval $run_command

#     sleep 1  # Adjust the sleep duration as needed
# done


# python ./LOO_nested_CV_train_skf.py jamba test_mlp_pre mdd_classification_jamba_light_data_mlp_preprocess_SKCV_4_holdout_5

# python ./LOO_nested_CV_train_skf.py jamba test mdd_classification_jamba_subject_110_hb_simple_all_1d_SPECIFY_FOLD_4_nor_raw

# python ./LOO_nested_CV_train_skf.py cnn_transformer retrain mdd_classification_mamba_subject_all_seq_ch_hb_simple_all_1d_SPECIFY_FOLD_3_holdout_4_nor

# python ./LOO_nested_CV_train_skf.py cnn_transformer eval_10 mdd_classification_all_hb_simple_all_1d_SPECIFY_FOLD_2_nor

# python ./LOO_nested_CV_train_skf.py jamba_hybrid eval_10 mdd_classification_jamba_hybrid_subject_all_hb_simple_all_1d_SPECIFY_FOLD_2_nor


# python ./LOO_nested_CV_train_skf.py jamba skf_40_hold_100_x mdd_classification_all_hb_simple_all_1d_SPECIFY_FOLD_40_holdout_200_nor

# python ./LOO_nested_CV_train.py jamba skf_5_loocv_ mdd_classification_all_hb_simple_all_1d_SPECIFY_FOLD_5_LOOCV