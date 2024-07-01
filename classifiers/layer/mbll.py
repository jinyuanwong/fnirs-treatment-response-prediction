import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, Lambda, Layer
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


    # e_HbO_704 = 298
    # e_HbR_704 = 1687.76
    # e_HbO_8279 = 965.2
    # e_HbR_8279 = 693.2
    
# Custom layer to apply MBLL with learnable parameters
class MBLL_Layer(Layer):
    def __init__(self, **kwargs):
        super(MBLL_Layer, self).__init__(**kwargs)
        # Initialize the extinction coefficients and DPF as learnable parameters
        self.epsilon_HbO_695 = self.add_weight(name='epsilon_HbO_695', shape=(), initializer=tf.keras.initializers.Constant(298), trainable=True)
        self.epsilon_HbR_695 = self.add_weight(name='epsilon_HbR_695', shape=(), initializer=tf.keras.initializers.Constant(1687.76), trainable=True)
        self.epsilon_HbO_830 = self.add_weight(name='epsilon_HbO_830', shape=(), initializer=tf.keras.initializers.Constant(965.2), trainable=True)
        self.epsilon_HbR_830 = self.add_weight(name='epsilon_HbR_830', shape=(), initializer=tf.keras.initializers.Constant(693.2), trainable=True)
        self.dpf_695 = self.add_weight(name='dpf_695', shape=(), initializer=tf.keras.initializers.Constant(3.0), trainable=True)
        self.dpf_830 = self.add_weight(name='dpf_830', shape=(), initializer=tf.keras.initializers.Constant(3.0), trainable=True)


    def call(self, inputs):
        
        # tf.print("Input shape:", tf.shape(inputs))  # Shape: (batch_size, channels, timesteps)
        
        timesteps_half = tf.shape(inputs)[-1] // 2

        I_695 = inputs[:, :, :timesteps_half]
        I_830 = inputs[:, :, timesteps_half:]
        # tf.print("I_695 shape:", tf.shape(I_695))  # Shape: (batch_size, channels, timesteps_half)
        # tf.print("I_830 shape:", tf.shape(I_830))  # Shape: (batch_size, channels, timesteps_half)

        # Compute the initial intensity as the mean of the first few timesteps
        baseline_period = 50  # Number of initial timesteps to consider as baseline
        I0_695 = tf.reduce_mean(I_695[:, :, :baseline_period], axis=-1, keepdims=True)
        I0_830 = tf.reduce_mean(I_830[:, :, :baseline_period], axis=-1, keepdims=True)
        # tf.print("I0_695 shape:", tf.shape(I0_695))  # Shape: (batch_size, channels, 1)
        # tf.print("I0_830 shape:", tf.shape(I0_830))  # Shape: (batch_size, channels, 1)

        OD_695 = -tf.math.log((I_695 + 1e-10) / (I0_695 + 1e-10))  # To avoid division by zero
        OD_830 = -tf.math.log((I_830 + 1e-10) / (I0_830 + 1e-10))
        # tf.print("OD_695 shape:", tf.shape(OD_695))  # Shape: (batch_size, channels, timesteps_half)
        # tf.print("OD_830 shape:", tf.shape(OD_830))  # Shape: (batch_size, channels, timesteps_half)

        # Construct the matrix E with the current extinction coefficients and DPFs
        E = tf.stack([
            [self.epsilon_HbO_695 * self.dpf_695, self.epsilon_HbR_695 * self.dpf_695],
            [self.epsilon_HbO_830 * self.dpf_830, self.epsilon_HbR_830 * self.dpf_830]
        ], axis=0)
        # tf.print("E shape:", tf.shape(E))  # Shape: (2, 2)

        # Ensure the matrix E is invertible
        E += tf.eye(2) * 1e-5  # Add a small identity matrix to ensure invertibility

        E_inv = tf.linalg.inv(E)
        # tf.print("E_inv shape:", tf.shape(E_inv))  # Shape: (2, 2)

        delta_OD = tf.stack([OD_695, OD_830], axis=-1)
        # tf.print("delta_OD shape:", tf.shape(delta_OD))  # Shape: (batch_size, channels, timesteps_half, 2)

        delta_OD = tf.reshape(delta_OD, (-1, 2))
        # tf.print("delta_OD reshaped shape:", tf.shape(delta_OD))  # Shape: (batch_size * channels * timesteps_half, 2)

        delta_Hb = tf.matmul(delta_OD, E_inv)
        # tf.print("delta_Hb shape:", tf.shape(delta_Hb))  # Shape: (batch_size * channels * timesteps_half, 2)

        delta_Hb = tf.reshape(delta_Hb, (-1, inputs.shape[1], timesteps_half, 2))
        # tf.print("delta_Hb reshaped shape:", tf.shape(delta_Hb))  # Shape: (batch_size, channels, timesteps_half, 2)
        
        return delta_Hb
