from configs.config import *
from configs.models_args.mamba_args import ModelArgs
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import math
import os

class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, **kwargs):
        super(CustomLearningRateSchedule, self).__init__()
        self.max_lr = kwargs.get('max_lr', 1e-4)
        self.warmup_step = kwargs.get('warmup_step', 4000)
        self.total_step = kwargs.get('total_step', 20000)
        self.decay_step = self.total_step - self.warmup_step
        self.end_lr = kwargs.get('end_lr', 1e-8) 

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        def warmup_fn():
            return self.max_lr * (step / self.warmup_step)

        def decay_fn():
            decay_step = step - self.warmup_step
            decay_rate = (self.end_lr / self.max_lr) ** (1 / self.decay_step)
            return self.max_lr * (decay_rate ** decay_step)

        return tf.cond(step < self.warmup_step, warmup_fn, decay_fn)




# Instantiate the custom learning rate schedule
# learning_rate_schedule = CustomLearningRateSchedule(max_lr, warmup_step, total_step)


class Jamba_ModelArgs_extend_from_Mamba(ModelArgs):
    n_heads: int = 4
    transformers_layers: int = 3
    n_experts: int = 4 # for MoE
    global_pooling: bool = False
    # lr_scheduler = LearningRateScheduler(sinusoidal_lr)

    def __post_init__(self):
        super().__post_init__() 
        self.learning_rate = CustomLearningRateSchedule(warmup_step=self.warmup_step, end_lr=1e-8)
        self.earlystopping = EarlyStopping(monitor='val_loss', patience=self.patiences)
    
