

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from classifiers.layer.transformers import MultiHeadAttention
from classifiers.layer.rmsnorm import RMSNorm
from classifiers.layer.moe import MoE
from classifiers.layer.mamba import MambaBlock


class Transformer_layer(layers.Layer):
    def __init__(self,
                 args):
        super(Transformer_layer, self).__init__()

        self.rms_norm1 = RMSNorm()
        self.attention = MultiHeadAttention(args.n_heads)
        self.rms_norm2 = RMSNorm()
        self.mlp = layers.Dense(args.model_internal_dim, activation=args.activation, kernel_regularizer=keras.regularizers.l2(args.l2_rate))
        self.dropout = layers.Dropout(args.dropout_rate)

    def call(self, inputs):
        x = self.rms_norm1(inputs)
        x = self.attention(x, x, x)
        x1 = x + inputs
        x = self.rms_norm2(x1)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + x1
        return x


class Attention_MoE_layer(layers.Layer):
    def __init__(self,
                 args):
        super(Attention_MoE_layer, self).__init__()

        self.rms_norm1 = RMSNorm()
        self.attention = MultiHeadAttention(args.n_heads)
        self.rms_norm2 = RMSNorm()
        self.moe = MoE(args)
        self.dropout = layers.Dropout(args.dropout_rate)

    def call(self, inputs):
        x = self.rms_norm1(inputs)
        x = self.attention(x, x, x)
        x1 = x + inputs
        x = self.rms_norm2(x1)
        x = self.moe(x)
        x = self.dropout(x)
        x = x + x1
        return x


class Mamba_layer(layers.Layer):
    def __init__(self,
                 args):
        super(Mamba_layer, self).__init__()

        self.rms_norm1 = RMSNorm()
        self.mamba = MambaBlock(args)
        self.rms_norm2 = RMSNorm()
        self.mlp = layers.Dense(
            args.model_internal_dim, activation=args.activation, kernel_regularizer=keras.regularizers.l2(args.l2_rate))

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
                 args):
        super(Mamba_MoE_layer, self).__init__()

        self.rms_norm1 = RMSNorm()
        self.mamba = MambaBlock(args)
        self.rms_norm2 = RMSNorm()
        self.moe = MoE(args)

    def call(self, inputs):
        x = self.rms_norm1(inputs)
        x = self.mamba(x)
        x1 = x + inputs
        x = self.rms_norm2(x1)
        x = self.moe(x)
        x = x + x1
        return x
