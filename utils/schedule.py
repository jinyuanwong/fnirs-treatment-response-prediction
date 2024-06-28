import tensorflow as tf
import numpy as np

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