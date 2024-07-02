import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, Lambda, Layer
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from classifiers.layer.rmsnorm import RMSNorm
    # e_HbO_704 = 298
    # e_HbR_704 = 1687.76
    # e_HbO_8279 = 965.2
    # e_HbR_8279 = 693.2
    
# Custom layer to apply MBLL with learnable parameters
class MBLL_Layer(Layer):
    def __init__(self, baseline_period=5, **kwargs):
        baseline_period = kwargs.pop('baseline_period', 50)  # Extract baseline_period from kwargs with a default value        
        super(MBLL_Layer, self).__init__(**kwargs)
        # Initialize the extinction coefficients and DPF as learnable parameters
        self.epsilon_HbO_695 = self.add_weight(name='epsilon_HbO_695', shape=(), initializer=tf.keras.initializers.Constant(279.2), trainable=True) # 279.2, 298
        self.epsilon_HbR_695 = self.add_weight(name='epsilon_HbR_695', shape=(), initializer=tf.keras.initializers.Constant(1949.04), trainable=True) # 1949.04 1687.76
        self.epsilon_HbO_830 = self.add_weight(name='epsilon_HbO_830', shape=(), initializer=tf.keras.initializers.Constant(974), trainable=True)
        self.epsilon_HbR_830 = self.add_weight(name='epsilon_HbR_830', shape=(), initializer=tf.keras.initializers.Constant(693.04), trainable=True)
        # self.dpf_695 = self.add_weight(name='dpf_695', shape=(), initializer=tf.keras.initializers.Constant(2.5), trainable=True)
        # self.dpf_830 = self.add_weight(name='dpf_830', shape=(), initializer=tf.keras.initializers.Constant(2.5), trainable=True)
        self.dpf_695_unconstrained = self.add_weight(name='dpf_695_unconstrained', shape=(), initializer=tf.keras.initializers.Constant(0.5), trainable=True)
        self.dpf_830_unconstrained = self.add_weight(name='dpf_830_unconstrained', shape=(), initializer=tf.keras.initializers.Constant(0.5), trainable=True)        
        self.baseline_period = baseline_period

    def call(self, inputs):
        # Transform unconstrained weights to desired range [1.5, 3.5]
        self.dpf_695 = 1.5 + tf.sigmoid(self.dpf_695_unconstrained) * 2.0
        self.dpf_830 = 1.5 + tf.sigmoid(self.dpf_830_unconstrained) * 2.0
                
        # tf.print("Input shape:", tf.shape(inputs))  # Shape: (batch_size, channels, timesteps)
        timesteps_half = tf.shape(inputs)[-1] // 2

        I_695 = inputs[:, :, :timesteps_half]
        I_830 = inputs[:, :, timesteps_half:]
        # tf.print("I_695 shape:", tf.shape(I_695))  # Shape: (batch_size, channels, timesteps_half)
        # tf.print("I_830 shape:", tf.shape(I_830))  # Shape: (batch_size, channels, timesteps_half)

        # Compute the initial intensity as the mean of the first few timesteps
        baseline_period = self.baseline_period # tf.shape(inputs)[-1]  # Number of initial timesteps to consider as baseline
        I0_695 = tf.reduce_mean(I_695[:, :, :baseline_period], axis=-1, keepdims=True)
        I0_830 = tf.reduce_mean(I_830[:, :, :baseline_period], axis=-1, keepdims=True)
        # tf.print("I0_695 shape:", tf.shape(I0_695))  # Shape: (batch_size, channels, 1)
        # tf.print("I0_830 shape:", tf.shape(I0_830))  # Shape: (batch_size, channels, 1)

        OD_695 = -tf.math.log((I_695 + 1e-10) / (I0_695 + 1e-10))  # To avoid division by zero
        OD_830 = -tf.math.log((I_830 + 1e-10) / (I0_830 + 1e-10))
        # tf.print("OD_695 shape:", tf.shape(OD_695))  # Shape: (batch_size, channels, timesteps_half)
        # tf.print("OD_830 shape:", tf.shape(OD_830))  # Shape: (batch_size, channels, timesteps_half)

        # Subtract the baseline OD
        # baseline_OD_695 = tf.reduce_mean(OD_695[:, :, :baseline_period], axis=-1, keepdims=True)
        # baseline_OD_830 = tf.reduce_mean(OD_830[:, :, :baseline_period], axis=-1, keepdims=True)

        # delta_OD_695 = OD_695 - baseline_OD_695
        # delta_OD_830 = OD_830 - baseline_OD_830
        
        delta_OD_695 = OD_695
        delta_OD_830 = OD_830    

        # Construct the matrix E with the current extinction coefficients and DPFs
        E = tf.stack([
            [self.epsilon_HbO_695 * self.dpf_695, self.epsilon_HbR_695 * self.dpf_695],
            [self.epsilon_HbO_830 * self.dpf_830, self.epsilon_HbR_830 * self.dpf_830]
        ], axis=0)
        # tf.print("E shape:", tf.shape(E))  # Shape: (2, 2)

        # Print the values of E
        # tf.print("E values:")
        # tf.print(E)
        
        # Ensure the matrix E is invertible
        E += tf.eye(2) * 1e-5  # Add a small identity matrix to ensure invertibility

        E_inv = tf.linalg.inv(E)
        # tf.print("E_inv shape:", tf.shape(E_inv))  # Shape: (2, 2)

        delta_OD = tf.stack([delta_OD_695, delta_OD_830], axis=-1)
        # tf.print("delta_OD shape:", tf.shape(delta_OD))  # Shape: (batch_size, channels, timesteps_half, 2)

        delta_OD = tf.reshape(delta_OD, (-1, 2))
        # tf.print("delta_OD reshaped shape:", tf.shape(delta_OD))  # Shape: (batch_size * channels * timesteps_half, 2)

        delta_Hb = tf.matmul(delta_OD, E_inv)
        # tf.print("delta_Hb shape:", tf.shape(delta_Hb))  # Shape: (batch_size * channels * timesteps_half, 2)

        delta_Hb = tf.reshape(delta_Hb, (-1, inputs.shape[1], timesteps_half, 2))
        # tf.print("delta_Hb reshaped shape:", tf.shape(delta_Hb))  # Shape: (batch_size, channels, timesteps_half, 2)
        
        return delta_Hb

""" 
Args :

inputs: Subejcts, Channel, Timepoints(first half is 695, second half is 830)

outputs: Subjects, Channel, Timepoints, 2 (HbO, HbR)

"""
class MLP_Preprocess_Layer(Layer):
    def __init__(self, units1=5, units2=5, units3=5, units4=2, **kwargs):
        super(MLP_Preprocess_Layer, self).__init__(**kwargs)
        self.units1 = units1
        self.units2 = units2
        self.units3 = units3
        self.units4 = units4
        self.dense1 = Dense(self.units1)
        self.dense2 = Dense(self.units2)
        self.dense3 = Dense(self.units3)
        self.dense4 = Dense(self.units4)
        self.norm = RMSNorm()
    def call(self, inputs):
        
        # tf.print("Input shape:", tf.shape(inputs))  # Shape: (batch_size, channels, timesteps)
        timesteps_half = tf.shape(inputs)[-1] // 2
        
        I_695 = inputs[:, :, :timesteps_half]
        I_830 = inputs[:, :, timesteps_half:]
        
        inputs = tf.stack((I_695, I_830), axis=-1)
        inputs = self.norm(inputs)
        # Apply dense layers
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        outputs = self.dense4(x)
        
        # Reshape outputs
        outputs_reshaped = tf.reshape(outputs, (-1, inputs.shape[1], timesteps_half, 2))
        
        return outputs_reshaped