
from configs.config import *
import os
from configs.models_args.jamba_args import Jamba_ModelArgs_extend_from_Mamba

INPUT_HB_TYPE = ['diagnosis514']
SPECIFY_FOLD = 10
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 3500
HOLD_OUT_DIV = 10

args = Jamba_ModelArgs_extend_from_Mamba(
            batch_size=64,
            classweight1=1,
            patiences=300,
            lr_begin = 1e7, # 1e7 -> 1e5
            model_input_dims=128,
            model_states=64,# 64 -> 128
            last_dense_units=64,
            num_layers=2, # 2 -> 1
            dropout_rate=0.35, # 0.35 -> 0.15
            vocab_size=2,
            num_classes=2,
            warmup_step = 4000,
            loss='categorical_crossentropy', # 'binary_crossentropy', # categorical_crossentropy
            projection_expand_factor=1,
        )        

PARAMETER['jamba'] = {
    'hb_path': 'hbo_simple_data.npy',
    'args': args,
    'config_file_path': os.path.abspath(__file__),
}

PARAMETER['jamba_multimodal'] = {
    'hb_path': 'hb_simple_3d.npy',
    'args': args,
    'config_file_path': os.path.abspath(__file__),
}


# python ./LOO_nested_CV_train_skf.py jamba_multimodal skf_t1_test_conv1d  mdd_classification_jamba 



# refer https://towardsdatascience.com/mamba-ssm-theory-and-implementation-in-keras-and-tensorflow-32d6d4b32546
# PARAMETER['mamba'] = {
#     'hb_path': 'hb_data.npy',
#     'classweight1': 1, # 20240613: 40 -> 100 
#     'batch_size': 128, # choosing 8 because there are 8 in validation set.
#     'lr_begin': 100000, # 20240613: 0.0001 -> 0.00001
#     'model_input_dims': 128,
#     'model_states': 64,
#     'projection_expand_factor': 2,
#     'conv_kernel_size': 4,
#     'delta_t_min': 0.001,
#     'delta_t_max': 0.1,
#     'delta_t_scale': 0.1,
#     'delta_t_init_floor': 1e-4,
#     'conv_use_bias': True,
#     'dense_use_bias': False,
#     'layer_id': -1,
#     'seq_length': 128,
#     'num_layers': 5, # 5->3
#     'dropout_rate': 0.5, # 0.2->0.5
#     'use_lm_head': False,
#     'num_classes': None,
#     'vocab_size': None,
#     'final_activation': None
# }

# python ./LOO_nested_CV_train.py mamba loocv_v0 pretreatment_response_mamba