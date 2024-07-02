from dataclasses import dataclass, field
from typing import Tuple, List, Dict

import tensorflow as tf
from tensorflow import keras

from utils.schedule import CustomLearningRateSchedule

@dataclass
class Transformer_ModelArgs:
    batch_size: int
    d_model: int
    n_heads: int
    dropout_rate: float = 0.4
    n_layers: int = 6
    FFN_units: int = 256
    activation: str = 'gelu'
    final_activation: str = 'softmax'
    warmup_step: int = 4000
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.999
    num_classes: int = 2
    l2_rate: float = 0.001
    num_class: int = 2
    class_weights: Dict[int, float] = field(default_factory=lambda: {0: 1, 1: 1})
    kernel_size_1: Tuple[int, int] = (4, 5)
    stride_size_1: Tuple[int, int] = (1, 2)
    kernel_size_2: Tuple[int, int] = (1, 5)
    stride_size_2: Tuple[int, int] = (1, 2)
    output_channel: int = 4
    
    lr_begin: int = 100000
    warmup_step: int = 4000
    beta_1 = 0.9
    beta_2 = 0.99
    epsilon = 1e-9    
    
    earlystopping: keras.callbacks.EarlyStopping = None
    monitor_metric_early_stop: str = 'val_loss'
    patience: int = 300

    def __post_init__(self):
        self.kernel_size = [self.kernel_size_1, self.kernel_size_2]
        self.stride_size = [self.stride_size_1, self.stride_size_2]
        
        self.learning_rate = CustomLearningRateSchedule(warmup_step=self.warmup_step, end_lr=1e-8)
        
        self.earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor=self.monitor_metric_early_stop, patience=self.patience, restore_best_weights=True
        )
        
        if self.num_classes == 1:
            self.final_activation = 'sigmoid'
        else:
            self.final_activation = 'softmax'
            
    @property
    def optimizer(self):
        return tf.keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            beta_1=self.adam_beta_1,
            beta_2=self.adam_beta_2,
            epsilon=self.epsilon
        )            