


# 'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_64_model_states_1024'
# 'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_64_model_states_16'
# 'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_64_model_states_32'
model='jamba_MTL'
config_names=(
'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_64_model_states_64'
'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_64_model_states_128'
'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_64_model_states_256'
'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_64_model_states_512'
)
# 'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_16'
# 'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_32'
# 'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_64'
# 'MTL_all_hb_simple_all_1d_NCV_nor_STL_depression_AUG_0_layers_0_input_dims_256'

# 'MTL_all_hb_simple_all_1d_SPECIFY_FOLD_4_holdout_5_nor_loss'


itr_name='MTL_20240710_VF3_AugmentRatio'
seeds=(31415926 27182818 16180339 12345678 98765432)
# python_file="./LOO_nested_CV_train_skf.py"
python_file="./nested_CV_train.py"