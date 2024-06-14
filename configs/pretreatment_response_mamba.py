from configs.config import *


INPUT_HB_TYPE = ['prognosis_mix_hb/pretreatment_response']
SPECIFY_FOLD = 5
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 1000

# refer https://towardsdatascience.com/mamba-ssm-theory-and-implementation-in-keras-and-tensorflow-32d6d4b32546
PARAMETER['mamba'] = {
    'hb_path': 'hb_data.npy',
    'classweight1': 100, # 20240613: 40 -> 100 
    'batch_size': 8, # choosing 8 because there are 8 in validation set.
    'lr_begin': 100000,
    'model_input_dims': 128,
    'model_states': 64,
    'projection_expand_factor': 2,
    'conv_kernel_size': 4,
    'delta_t_min': 0.001,
    'delta_t_max': 0.1,
    'delta_t_scale': 0.1,
    'delta_t_init_floor': 1e-4,
    'conv_use_bias': True,
    'dense_use_bias': False,
    'layer_id': -1,
    'seq_length': 128,
    'num_layers': 3, # 5->3
    'dropout_rate': 0.5, # 0.2->0.5
    'use_lm_head': False,
    'num_classes': None,
    'vocab_size': None,
    'final_activation': None
}

# python ./LOO_nested_CV_train.py mamba loocv_v0 pretreatment_response_mamba