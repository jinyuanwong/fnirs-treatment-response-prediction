import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, Layer
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


class ChannelSelectionLayer(Layer):
    def __init__(self, timesteps, channels, **kwargs):
        super(ChannelSelectionLayer, self).__init__(**kwargs)
        # Initialize the channel mask as a trainable parameter
        self.timesteps = timesteps
        self.channels = channels
        self.channel_mask = self.add_weight(name='channel_mask', shape=(timesteps,), initializer='uniform', trainable=True)

    def call(self, inputs):
        # Expand the mask to match the input shape
        mask = tf.expand_dims(self.channel_mask, axis=0)
        mask = tf.expand_dims(mask, axis=0)
        # Mask shape: (1, 1, timesteps)
        
        # Apply the mask to the input
        masked_inputs = inputs * mask
        # masked_inputs shape: (batch_size, channels, timesteps)
        
        # Compute the mean of the masked inputs over the timesteps
        mean_masked_inputs = tf.reduce_mean(masked_inputs, axis=-1)
        # mean_masked_inputs shape: (batch_size, channels)
        
        # Compute the sum over all subjects to rank the channels
        channel_scores = tf.reduce_sum(mean_masked_inputs, axis=0)
        # channel_scores shape: (channels,)
        
        # Select the top channels
        top_k = self.channels // 2  # Select the top half
        _, top_indices = tf.nn.top_k(channel_scores, k=top_k)
        
        # Gather the top channels
        selected_inputs = tf.gather(inputs, top_indices, axis=1)
        # selected_inputs shape: (batch_size, top_k, timesteps)
        
        return selected_inputs
    
# class ChannelSelectionLayer(Layer):
#     def __init__(self, channels, **kwargs):
#         super(ChannelSelectionLayer, self).__init__(**kwargs)
#         # Initialize the channel mask as a trainable parameter
#         self.channels = channels
#         self.channel_mask = self.add_weight(name='channel_mask', shape=(channels,), initializer='uniform', trainable=True)

#     def call(self, inputs):
#         # Expand the mask to match the input shape
#         mask = tf.expand_dims(self.channel_mask, axis=0)
#         mask = tf.expand_dims(mask, axis=-1)
#         # tf.print("Channel mask:", self.channel_mask)

#         # Sort the channels based on the mask values
#         top_k = tf.shape(mask)[1] //4 * 3  # Select the top half
#         _, top_indices = tf.nn.top_k(self.channel_mask, k=top_k)

#         # Gather the top channels
#         selected_inputs = tf.gather(inputs, top_indices, axis=1)
#         # tf.print("Selected inputs shape:", tf.shape(selected_inputs))
        
#         return selected_inputs