from dataclasses import dataclass, field
from typing import Tuple, List, Dict
import tensorflow as tf
from tensorflow import keras
from utils.schedule import CustomLearningRateSchedule
from configs.models_args.model_args import MotherArgs
from utils.callbacks import reduceLRonplateau

@dataclass
class Transformer_ModelArgs(MotherArgs):

    # Transformer parameter settings
    d_model: int = 64
    n_heads: int = 1
    n_layers: int = 3
    FFN_units: int = 256
    class_weights: Dict[int, float] = field(default_factory=lambda: {0: 1, 1: 1})
    kernel_size_1: Tuple[int, int] = (4, 5)
    stride_size_1: Tuple[int, int] = (1, 2)
    kernel_size_2: Tuple[int, int] = (1, 5)
    stride_size_2: Tuple[int, int] = (1, 2)
    output_channel: int = 4
    
    lr_begin: int = 100000
    warmup_step: int = 4000
    
    def __post_init__(self):
        self.kernel_size = [self.kernel_size_1, self.kernel_size_2]
        self.stride_size = [self.stride_size_1, self.stride_size_2]
        
        # set learning rate
        if self.warmup_step == None:
            raise ValueError(f'warmup_step cannot be {self.warmup_step}')
        else:
            self.set_learning_rate(self.lr_begin, self.warmup_step, mode='CustomLearningRateSchedule')
        
        # set final_activation
        if self.num_classes == None:
            raise ValueError(f'num classes cannot be {self.num_classes}')
        else:
            self.set_final_activation()        
        self.reduce_lr = reduceLRonplateau()
 