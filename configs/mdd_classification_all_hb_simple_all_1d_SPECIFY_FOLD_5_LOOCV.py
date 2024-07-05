from configs.config import *
from configs.models_args.transformer_args import Transformer_ModelArgs
import os 

from configs.mdd_classification_jamba import *
INPUT_HB_TYPE = ['diagnosis514']
SPECIFY_FOLD = 5
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 1000
# HOLD_OUT_DIV = 200

transformer_args = Transformer_ModelArgs(
    batch_size=64,
    d_model=64,
    n_heads=1,
    class_weights={0: 1, 1: 1}, # for pretreament classification {0: 1, 1: 5}
    )

        
PARAMETER['cnn_transformer'] = {
    'hb_path': 'nor_hb_simple_all_1d.npy',
    'args': transformer_args,
}
args = Jamba_ModelArgs_extend_from_Mamba(
            batch_size= 512,
            classweight1=1,
            patiences= 5,
            lr_begin = 1e7, # 1e7 -> 1e5
            model_input_dims=128,
            model_states=64,# 64 -> 128
            last_dense_units=64,
            num_layers=2, # 2 -> 1
            dropout_rate=0.5, # 0.35 -> 0.15
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


for model, val in PARAMETER.items():
    PARAMETER[model]['hb_path'] = 'nor_hb_simple_all_1d.npy'#'nor_hb_simple_all_1d.npy'
    PARAMETER[model]['classweight1'] = 1
    