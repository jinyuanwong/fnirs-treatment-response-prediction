import tensorflow as tf
from tensorflow.keras import layers

class RMSNorm(layers.Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super(RMSNorm, self).__init__(**kwargs)
        self.epsilon = epsilon 
        
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer='ones', trainable=True)
        
    # def call(self, x):
    #     # this is wrong because you average over all dimensions, which should be only operated in last dimension
    #     return x / tf.sqrt(tf.reduce_mean(tf.square(x))) * self.gamma
    
    def call(self, x):
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.epsilon)
        return self.gamma * x / rms    

