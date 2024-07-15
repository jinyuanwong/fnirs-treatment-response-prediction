from configs.config import *
from configs.models_args.mamba_args import ModelArgs
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import math
import os
from utils.schedule import CustomLearningRateSchedule
from utils.callbacks import reduceLRonplateau
class Jamba_ModelArgs_extend_from_Mamba(ModelArgs):
    
    n_heads: int = 4
    transformers_layers: int = 3
    n_experts: int = 4 # for MoE
    global_pooling: bool = False
    l2_rate: float = 0.001
    # lr_scheduler = LearningRateScheduler(sinusoidal_lr)

    # For jamba_MTL_V (V>V2)
    use_mlp_layer: bool = True #
    use_gnn_layer: bool = True #
    use_conv1d_layer: bool = True #
    use_mamba_block: bool = True #

    def __post_init__(self):
        super().__post_init__() 

        if self.warmup_step == None:
            raise ValueError(f'warmup_step cannot be {self.warmup_step}')
        else:
            self.set_learning_rate(self.lr_begin, self.warmup_step, mode='CustomLearningRateSchedule')
            
        # self.earlystopping = EarlyStopping(monitor=self.monitor_metric_early_stop, patience=self.patiences)
        self.reduce_lr = reduceLRonplateau()
    
