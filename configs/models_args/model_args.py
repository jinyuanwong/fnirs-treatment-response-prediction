from dataclasses import dataclass, field
from typing import Union
import tensorflow as tf
import tensorflow.keras as keras
from utils.utils_mine import CustomSchedule
from utils.schedule import CustomLearningRateSchedule
from classifiers.layer.rmsnorm import RMSNorm
@dataclass
class MotherArgs:
    # normal model parameters
    dropout_rate: float = 0.3
    activation: callable = tf.nn.gelu
    num_classes: int = 2
    final_activation: str = 'softmax'
    l2_rate: float = 0.01
    load_previous_checkpoint: bool = True
    normalization_method: str = 'RMSNorm' # 'RMSNorm' 'BatchNormalization' 'LayerNormalization'
    
    # model.fit parameters
    batch_size: int = 8
    epochs: int = 100
    callbacks: list = None
    verbose: bool = True
    shuffle: bool = True
    
    # model.complie parameters
    # learning rate parameters
    # optimizer: keras.optimizers.Optimizer = None
    learning_rate: Union[float, keras.optimizers.schedules.LearningRateSchedule] = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.99
    epsilon: float = 1e-9
    clipping_method: str = 'clipnorm'
    clipvalue: float = 1.0
    clipnorm: float = 1.0
    weight_decay: float = 0.004
    lr_mode: str = 'CustomLearningRateSchedule' # 'CustomSchedule' 'constant' 'CustomLearningRateSchedule'
    lr_begin: int = 100000
    lr_first_decay_steps: int = 15 # for CosineDecayRestarts
    warmup_step: int = 4000
    
    # model.complie parameters
    # these two the metrics for model compile, which will be shown during fitting
    # very important, especially for classfication and Multi-task learning classification    
    loss: Union[str, keras.losses.Loss, dict] = None
    metrics: dict = None 
    
    # Early stopping parameters
    # earlystopping: keras.callbacks.EarlyStopping = None
    monitor_metric_early_stop: str = 'val_loss'
    patiences: int = 15

    # Checkpoint parameters 
    checkpoint_path: str = None
    monitor_metric_mode: str = 'min'
    monitor_metric_checkpoint: str = 'val_loss'
    delete_checkpoint: bool = True
    
    # reduce_lr is not working iif lr is a schedule
    reduce_lr: keras.callbacks.ReduceLROnPlateau = None
    
    def update_model_checkpoint(self, checkpoint_path):
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path
            self.model_checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_path,
                monitor=self.monitor_metric_checkpoint,
                mode=self.monitor_metric_mode,
                save_weights_only=True,
                save_best_only=True,
            )
        else:
            self.model_checkpoint = None
        
    def set_learning_rate(self):
        # print(f"set_learning_rate -> lr_begin: {lr_begin}" )
        
        lr_begin = self.lr_begin
        warmup_step = self.warmup_step
        mode = self.lr_mode
        
        if mode == 'CustomSchedule':
            self.learning_rate = CustomSchedule(lr_begin, warmup_step)
        elif mode == 'CustomLearningRateSchedule':
            self.learning_rate = CustomLearningRateSchedule(warmup_step=warmup_step)
        elif mode == 'constant':
            self.learning_rate = lr_begin
        elif mode == 'CosineDecayRestarts':
            initial_learning_rate = lr_begin
            first_decay_steps = self.lr_first_decay_steps
            t_mul = 2.0
            m_mul = 0.8
            alpha = 0.0
            learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=initial_learning_rate,
                first_decay_steps=first_decay_steps,
                t_mul=t_mul,
                m_mul=m_mul,
                alpha=alpha,
                name='SGDRDecay'
            )
            self.learning_rate = learning_rate_schedule
        else:
            raise ValueError(f"mode {mode} not supported in set_learning_rate")
    
    def set_final_activation(self):
        if self.num_classes == 1:
            self.final_activation = 'sigmoid'
        else:
            self.final_activation = 'softmax'
        
    
    @property
    def optimizer(self, mode='adamw'):
        
        clipping_method = self.clipping_method
        self.set_learning_rate()
        if mode == 'adamw':
            if clipping_method == 'clipnorm':
                return keras.optimizers.AdamW(self.learning_rate,
                                              beta_1=self.beta_1,
                                              beta_2=self.beta_2,
                                              epsilon=self.epsilon,
                                              clipnorm=self.clipnorm,
                                              weight_decay=self.weight_decay)
            elif clipping_method == 'clipvalue':
                return keras.optimizers.AdamW(self.learning_rate,
                                              beta_1=self.beta_1,
                                              beta_2=self.beta_2,
                                              epsilon=self.epsilon,
                                              clipvalue=self.clipvalue,
                                              weight_decay=self.weight_decay)
            else:
                raise ValueError(f"clipping_method {clipping_method} not supported")
        else:
            raise ValueError(f"mode {mode} not supported in optimizer")

    @property 
    def earlystopping(self):
        return keras.callbacks.EarlyStopping(
            monitor=self.monitor_metric_early_stop, patience=self.patiences)
        
    @property
    def normalization_layer(self):
        method = self.normalization_method
        if method == 'RMSNorm':
            return RMSNorm
        elif method == 'BatchNormalization':
            return keras.layers.BatchNormalization
        elif method == 'LayerNormalization':
            return keras.layers.LayerNormalization
        else:
            raise ValueError(f"normalization method {method} not supported")