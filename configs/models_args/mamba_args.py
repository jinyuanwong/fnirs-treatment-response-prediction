from configs.config import *
from dataclasses import dataclass
from typing import Union
import math
from utils.utils_mine import CustomSchedule

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping

import os 

@dataclass
class ModelArgs:
    batch_size: int = 8
    classweight1: int = 15
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
    
    lr_begin: int = 100000
    warmup_step: int = 4000
    beta_1 = 0.9
    beta_2 = 0.99
    epsilon = 1e-9    
    
    loss:Union[str, keras.losses.Loss] = None
    optimizer: keras.optimizers.Optimizer = None
    earlystopping: keras.callbacks.EarlyStopping = None
    metrics = ['accuracy']
    monitor_metric_early_stop: str = 'val_loss'


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

        if self.loss == None:
            raise ValueError(f"loss cannot be {self.loss}")
        
        if self.patiences == None:
            raise ValueError(f"patiences cannot be {self.patiences}")
        else:
            self.earlystopping = EarlyStopping(monitor=self.monitor_metric_early_stop, patience=self.patiences)
