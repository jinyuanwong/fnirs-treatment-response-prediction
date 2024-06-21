
import tensorflow as tf 
from tensorflow.keras import layers

class ChannelAttentionLayer(layers.Layer):
    def __init__(self, input_dim):
        super(ChannelAttentionLayer, self).__init__()
        self.input_dim = input_dim
        # Define a trainable 1D array (attention weights)
        self.attention_weights = self.add_weight(
            shape=(input_dim,), 
            initializer='uniform', 
            trainable=True, 
            dtype=tf.float32,
            name='ChannelAttentionLayer_attention_weights'
        )

    def call(self, inputs):
        # Diagonalize the 1D array to create a (52, 52) matrix
        diag_attention_weights = tf.linalg.diag(self.attention_weights)
        # Multiply the diagonal matrix with the input
        output = tf.matmul(diag_attention_weights, inputs)
        return output