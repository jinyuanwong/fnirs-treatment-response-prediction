
output_diretory = 'configs/config_seed'

num_of_seed_to_generate = 5000 

file_base_name = 'STL_depression_NCV_best_v1_wo_mamba_w_mlp_w_conv_model_seed_optimisation'


for seed in range(num_of_seed_to_generate):
    file_name = f'{file_base_name}_{seed}.py'
    content_in_the_file = f"""
from configs.STL_depression_NCV_best_v1_wo_mamba_w_mlp_w_conv_model_seed_optimisation import *
MODEL_SEED = {seed}
    """
    with open(f'{output_diretory}/{file_name}', 'w') as f:
        f.write(content_in_the_file)
    
    print(f'File {file_name} created')
