    # while true; do
    #     python scripts/automl/automl_main_all52_channels.py # scripts/ML/simple_prediction/clinical_data_modality_classification_v2_tuning_5_models_multiple_seeds.py #scripts/automl/automl_main_all52_channels.py
    #     sleep 1  # Adjust the sleep duration as needed
    # done



model='jamba'
config_file='mdd_classification_all_hb_simple_all_1d_SPECIFY_FOLD_5_LOOCV' #'mdd_classification_all_hb_simple_all_1d_SPECIFY_FOLD_4_holdout_5_nor'
itr_name='skf_5_loocv' #'eval'

# Loop from 0 to 9
for itr in {1..1000}
do
    # Construct the run_itr variable
    run_itr="$itr_name" #  $itr $itr$itr

    # Construct the run_command
    run_command="conda run -n tf python ./LOO_nested_CV_train_skf.py $model $run_itr $config_file"

    # Execute the run_command
    echo "Running command: $run_command"
    eval $run_command

    sleep 1  # Adjust the sleep duration as needed
done    


# python ./LOO_nested_CV_train.py jamba skf_5_loocv_ mdd_classification_all_hb_simple_all_1d_SPECIFY_FOLD_5_LOOCV