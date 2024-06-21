

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from classifiers.layer.transformers import MultiHeadAttention
from classifiers.layer.rmsnorm import RMSNorm
from classifiers.layer.moe import MoE
from classifiers.layer.mamba import MambaBlock


class Transformer_layer(layers.Layer):
    def __init__(self,
                 FFN_units,
                 n_heads,
                 dropout_rate,
                 activation):
        super(Transformer_layer, self).__init__()

        self.rms_norm1 = RMSNorm()
        self.attention = MultiHeadAttention(n_heads)
        self.rms_norm2 = RMSNorm()
        self.mlp = layers.Dense(FFN_units, activation=activation)

    def call(self, inputs):
        x = self.rms_norm1(inputs)
        x = self.attention(x, x, x)
        x1 = x + inputs
        x = self.rms_norm2(x1)
        x = self.mlp(x)
        x = x + x1
        return x


class Attention_MoE_layer(layers.Layer):
    def __init__(self,
                 FFN_units,
                 n_heads,
                 dropout_rate,
                 activation,
                 n_experts):
        super(Attention_MoE_layer, self).__init__()

        self.rms_norm1 = RMSNorm()
        self.attention = MultiHeadAttention(n_heads)
        self.rms_norm2 = RMSNorm()
        self.moe = MoE(n_experts, FFN_units, activation)

    def call(self, inputs):
        x = self.rms_norm1(inputs)
        x = self.attention(x, x, x)
        x1 = x + inputs
        x = self.rms_norm2(x1)
        x = self.moe(x)
        x = x + x1
        return x


class Mamba_layer(layers.Layer):
    def __init__(self,
                 modelargs):
        super(Mamba_layer, self).__init__()

        self.rms_norm1 = RMSNorm()
        self.mamba = MambaBlock(modelargs)
        self.rms_norm2 = RMSNorm()
        self.mlp = layers.Dense(
            modelargs.model_internal_dim, activation=modelargs.activation)

    def call(self, inputs):
        x = self.rms_norm1(inputs)
        x = self.mamba(x)
        x1 = x + inputs
        x = self.rms_norm2(x1)
        x = self.mlp(x)
        x = x + x1
        return x


class Mamba_MoE_layer(layers.Layer):
    def __init__(self,
                 modelargs):
        super(Mamba_MoE_layer, self).__init__()

        self.rms_norm1 = RMSNorm()
        self.mamba = MambaBlock(modelargs)
        self.rms_norm2 = RMSNorm()
        self.moe = MoE(modelargs.n_experts, modelargs.model_internal_dim, modelargs.activation)

    def call(self, inputs):
        x = self.rms_norm1(inputs)
        x = self.mamba(x)
        x1 = x + inputs
        x = self.rms_norm2(x1)
        x = self.moe(x)
        x = x + x1
        return x
