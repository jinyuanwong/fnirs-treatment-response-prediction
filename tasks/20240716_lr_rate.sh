

models=(
'jamba_MTL'
)
config_files=(
'NCV_STL_depression_AUG_0_layers_0_input_dims_128_model_states_128_size_512_modelStates_64_dims_512_lr_3e-5' 
'NCV_STL_depression_AUG_0_layers_0_input_dims_128_model_states_128_size_512_modelStates_64_dims_512_lr_5e-5' 
'NCV_STL_depression_AUG_0_layers_0_input_dims_128_model_states_128_size_512_modelStates_64_dims_512_lr_7e-5' 
'NCV_STL_depression_AUG_0_layers_0_input_dims_128_model_states_128_size_512_modelStates_64_dims_512_lr_9e-5' 
)
itr_name='jamba_20240716'
seeds=(31415926 27182818 16180339 12345678 98765432)
# python_file="./LOO_nested_CV_train_skf.py"
python_file="./nested_CV_train.py"


# 'NCV_STL_depression_AUG_0_layers_0_input_dims_128_model_states_128_size_512_modelStates_64_dims_512_lr_1e-3' 
# 'NCV_STL_depression_AUG_0_layers_0_input_dims_128_model_states_128_size_512_modelStates_64_dims_512_lr_1e-4' 
# 'NCV_STL_depression_AUG_0_layers_0_input_dims_128_model_states_128_size_512_modelStates_64_dims_512_lr_1e-5' 
# 'NCV_STL_depression_AUG_0_layers_0_input_dims_128_model_states_128_size_512_modelStates_64_dims_512_lr_1e-6' 
# 'NCV_STL_depression_AUG_0_layers_0_input_dims_128_model_states_128_size_512_modelStates_64_dims_512_lr_1e-7' 
# 'NCV_STL_depression_AUG_0_layers_0_input_dims_128_model_states_128_size_512_modelStates_64_dims_512_lr_2e-6' 
# 'NCV_STL_depression_AUG_0_layers_0_input_dims_128_model_states_128_size_512_modelStates_64_dims_512_lr_5e-6' 
# 'NCV_STL_depression_AUG_0_layers_0_input_dims_128_model_states_128_size_512_modelStates_64_dims_512_lr_5e-7' 