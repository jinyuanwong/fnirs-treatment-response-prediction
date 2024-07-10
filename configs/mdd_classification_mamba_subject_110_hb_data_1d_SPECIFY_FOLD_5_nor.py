from configs.config import *
from configs.models_args.mamba_args import ModelArgs
import os 
INPUT_HB_TYPE = ['diagnosis_110_fabeha_dataset_hb_all']
SPECIFY_FOLD = 5
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 1000
HOLD_OUT_DIV = 10

args = ModelArgs(
            batch_size=64,
            lr_begin = 1e7,
            classweight1 = 1,   
            patiences = 100,
            model_input_dims=128,
            model_states=64,
            last_dense_units=64,
            num_layers=3,
            dropout_rate=0.35,
            vocab_size=2,
            num_classes=2,
            warmup_step = 200,
            loss='categorical_crossentropy', # 'binary_crossentropy', # categorical_crossentropy
            projection_expand_factor=1,
        )        



PARAMETER['mamba'] = {
    'hb_path': 'nor_hb_data_1d.npy',
    'args': args,
    'config_file_path': [os.path.abspath(__file__)],

}