from configs.config import *
from configs.models_args.mamba_args import ModelArgs
import os 

INPUT_HB_TYPE = ['prognosis_mix_hb/pretreatment_response']
SPECIFY_FOLD = 5
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 1000



args = ModelArgs(
            batch_size = 8,
            lr_begin = 100000,
            classweight1 = 100,
            patiences = 100,
            conv_kernel_size=4,
            model_input_dims=128,
            model_states=64,
            last_dense_units=64,
            num_layers=3,
            dropout_rate=0.4,
            vocab_size=2,
            num_classes=2,
            warmup_step = 200,
            loss='categorical_crossentropy', # 'binary_crossentropy', # categorical_crossentropy
            projection_expand_factor=1,
        )        


PARAMETER['mamba'] = {
    'hb_path': 'hb_data.npy',
    'args': args,
    'config_file_path': [os.path.abspath(__file__)],
}


"""
this achieve good result loocv_v0 39 - 2
 accuracy: 1.0 | sensitivity: 1.0 | specificity: 1.0 | duration: 0 | F1-score: 1.0 | AUC: 1.0 || current_time_seed: 1719025704| message: loocv_v0| parameter: {'hb_path': 'hb_data.npy', 'args': ModelArgs(batch_size=8, lr_begin=100000, classweight1=100, last_dense_units=64, warmup_step=200, clipnorm=1.0, patiences=100, model_input_dims=128, model_states=64, projection_expand_factor=1, conv_kernel_size=1, delta_t_min=0.001, delta_t_max=0.1, delta_t_scale=0.1, delta_t_init_floor=0.0001, conv_use_bias=True, dense_use_bias=False, layer_id=242, seq_length=128, num_layers=3, dropout_rate=0.2, use_lm_head=False, num_classes=2, vocab_size=2, activation=<function gelu at 0x749b7eb98040>, loss='categorical_crossentropy', optimizer=None, earlystopping=<keras.src.callbacks.EarlyStopping object at 0x749ac06faac0>), 'config_file_path': '/home/jy/Documents/fnirs/treatment_response/fnirs-depression-deeplearning/configs/pretreatment_response_mamba.py'}| monitor_metric: accuracy

"""