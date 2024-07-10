
from configs.mdd_classification import *
import os
from configs.models_args.jamba_args import Jamba_ModelArgs_extend_from_Mamba
from configs.mdd_classification_jamba import *
INPUT_HB_TYPE = ['diagnosis514_light']
SPECIFY_FOLD = 4
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 3500
HOLD_OUT_DIV = 5

args = Jamba_ModelArgs_extend_from_Mamba(
            batch_size=64,
            classweight1=1,
            patiences=30,
            lr_begin = 1e3, # 1e7 -> 1e5
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
    'hb_path': 'light_data_mlp_preprocessed.npy',
    'args': args,
    'config_file_path': [os.path.abspath(__file__)],
}


for model, val in PARAMETER.items():
    PARAMETER[model]['hb_path'] = 'light_data_mlp_preprocessed.npy'
