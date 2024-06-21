from configs.config import *
from dataclasses import dataclass
from typing import Union
import math
from utils.utils_mine import CustomSchedule

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping


@dataclass
class ModelArgs:
    batch_size: int = 8
    lr_begin: int = 1e7
    classweight1: int = 1
    last_dense_units: int = 1024
    warmup_step: int = None
    clipnorm: float = 1.0
    
    patiences: int = 100
    model_input_dims: int = 128
    model_states: int = 32
    projection_expand_factor: int = 2
    conv_kernel_size: int = 1
    delta_t_min: float = 0.001 # not used 
    delta_t_max: float = 0.1 # not used 
    delta_t_scale: float = 0.1 # not used 
    delta_t_init_floor: float = 1e-4 # not used 
    conv_use_bias: bool = True
    dense_use_bias: bool = False
    layer_id: int = -1
    seq_length: int = 128
    num_layers: int = 5
    dropout_rate: float = 0.2
    use_lm_head: float = False
    num_classes: int = None
    vocab_size: int = 2
    activation: callable = tf.nn.gelu
    final_activation = None
    loss:Union[str, keras.losses.Loss] = None
    optimizer: keras.optimizers.Optimizer = None
    earlystopping: keras.callbacks.EarlyStopping = None
    metrics = ['accuracy']
    

    def __post_init__(self):
        self.model_internal_dim: int = int(self.projection_expand_factor * self.model_input_dims)

        self.delta_t_rank = math.ceil(self.model_input_dims/16)
        if self.layer_id == -1:
            self.layer_id = np.round(np.random.randint(0, 1000), 4)

        if self.vocab_size == None:
            raise ValueError("vocab size cannot be none")

        if self.use_lm_head:
            self.num_classes=self.vocab_size
        else:
            if self.num_classes == None:
                raise ValueError(f'num classes cannot be {self.num_classes}')

            if self.num_classes == 1:
                self.final_activation = 'sigmoid'
            else:
                self.final_activation = 'softmax'
        
        if self.warmup_step == None:
            raise ValueError(f'warmup_step cannot be {self.warmup_step}')
        else:
            self.learning_rate = CustomSchedule(self.lr_begin, self.warmup_step)
            self.beta_1 = 0.9
            self.beta_2 = 0.99
            self.epsilon = 1e-9

        
        if self.loss == None:
            raise ValueError(f"loss cannot be {self.loss}")
        
        if self.patiences == None:
            raise ValueError(f"patiences cannot be {self.patiences}")
        else:
            self.earlystopping = EarlyStopping(monitor='val_loss', patience=self.patiences)

INPUT_HB_TYPE = ['diagnosis514']
SPECIFY_FOLD = 10
STRATIFIED_CV_TOTAL_TRAININING_TIME = 5
MAX_EPOCHS = 1000
HOLD_OUT_DIV = 10

args = ModelArgs(
            batch_size=64,
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
    'hb_path': 'hb_data.npy',
    'args': args
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