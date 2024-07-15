from configs.config import *
from dataclasses import dataclass
from typing import Union
import math
from utils.utils_mine import CustomSchedule

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os 
from configs.models_args.model_args import MotherArgs
@dataclass
class ModelArgs(MotherArgs):

    
    # Mamba parameter settings
    classweight1: int = 15
    last_dense_units: int = 1024
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
    use_lm_head: float = False
    vocab_size: int = 2
    load_previous_checkpoint: bool = True

    # For jamba_MTL_V (V>V2)
    use_mlp_layer: bool = True #
    use_gnn_layer: bool = True #
    use_conv1d_layer: bool = True #
    use_mamba_block: bool = True #
        
    lr_begin: int = 100000
    warmup_step: int = 4000


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
            else:
                self.set_final_activation()
        
        if self.warmup_step == None:
            raise ValueError(f'warmup_step cannot be {self.warmup_step}')
        else:
            self.set_learning_rate(self.lr_begin, self.warmup_step, mode='CustomLearningRateSchedule')

        if self.loss == None:
            raise ValueError(f"loss cannot be {self.loss}")
        
