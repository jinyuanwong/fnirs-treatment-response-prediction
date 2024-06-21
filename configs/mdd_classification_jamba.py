from utils.utils_mine import CustomSchedule, sinusoidal_lr

from configs.config import *
from configs.mdd_classification_mamba import ModelArgs


from dataclasses import dataclass
from typing import Union

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler

import math
import os

class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, warmup_steps, total_steps, **kwargs):
        super(CustomLearningRateSchedule, self).__init__()
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_steps = total_steps - warmup_steps
        self.end_lr = kwargs.get('end_lr', 1e-8) 

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        def warmup_fn():
            return self.max_lr * (step / self.warmup_steps)

        def decay_fn():
            decay_step = step - self.warmup_steps
            decay_rate = (self.end_lr / self.max_lr) ** (1 / self.decay_steps)
            return self.max_lr * (decay_rate ** decay_step)

        return tf.cond(step < self.warmup_steps, warmup_fn, decay_fn)


# Parameters
max_lr = 1e-4
warmup_steps = 4000
total_steps = 20000

# Instantiate the custom learning rate schedule
# learning_rate_schedule = CustomLearningRateSchedule(max_lr, warmup_steps, total_steps)


class Jamba_ModelArgs_extend_from_Mamba(ModelArgs):
    n_heads: int = 4
    transformers_layers: int = 3
    n_experts: int = 4 # for MoE
    # lr_scheduler = LearningRateScheduler(sinusoidal_lr)



    def __post_init__(self):
        super().__post_init__() 
        self.learning_rate = CustomLearningRateSchedule(max_lr, warmup_steps, total_steps, end_lr=1e-8)
        print(f'self.patiences: {self.patiences}'*100)
        self.earlystopping = EarlyStopping(monitor='val_accuracy', patience=self.patiences)
    

INPUT_HB_TYPE = ['diagnosis514']
SPECIFY_FOLD = 10
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 3500
HOLD_OUT_DIV = 10

args = Jamba_ModelArgs_extend_from_Mamba(
            batch_size=64,
            patiences=300,
            lr_begin = 1e7, # 1e7 -> 1e5
            model_input_dims=128,
            model_states=64,# 64 -> 128
            last_dense_units=64,
            num_layers=2, # 2 -> 1
            dropout_rate=0.35, # 0.35 -> 0.15
            vocab_size=2,
            num_classes=2,
            warmup_step = 200,
            loss='categorical_crossentropy', # 'binary_crossentropy', # categorical_crossentropy
            projection_expand_factor=1,
        )        

PARAMETER['jamba'] = {
    'hb_path': 'hbo_simple_data.npy',
    'args': args,
    'config_file_path': os.path.abspath(__file__),
}




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