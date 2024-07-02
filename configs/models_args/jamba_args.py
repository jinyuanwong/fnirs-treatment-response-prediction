from configs.config import *
from configs.models_args.mamba_args import ModelArgs
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import math
import os
from utils.schedule import CustomLearningRateSchedule

class Jamba_ModelArgs_extend_from_Mamba(ModelArgs):
    n_heads: int = 4
    transformers_layers: int = 3
    n_experts: int = 4 # for MoE
    global_pooling: bool = False
    l2_rate: float = 0.01
    # lr_scheduler = LearningRateScheduler(sinusoidal_lr)

    def __post_init__(self):
        super().__post_init__() 
        self.learning_rate = CustomLearningRateSchedule(warmup_step=self.warmup_step, end_lr=1e-8)
        self.earlystopping = EarlyStopping(monitor=self.monitor_metric_early_stop, patience=self.patiences)
    
