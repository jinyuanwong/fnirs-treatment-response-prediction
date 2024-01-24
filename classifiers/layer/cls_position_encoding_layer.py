
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.optimizers as ko
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow.keras as keras
import tensorflow_addons as tfa

import pandas as pd
import math
from tensorflow.keras.metrics import Recall
from tensorflow.keras.callbacks import EarlyStopping


class PositionalEncoding(layers.Layer):

    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_angles(self, pos, i, d_model):  # pos: (seq_length, 1) i: (1, d_model)
        # 2*(i//2) => if i = 5 -> ans = 4
        angles = 1 / np.power(10000., (2*(i//2)) / np.float32(d_model))
        return pos * angles  # (seq_length, d_model)

    def call(self, inputs):
        # input shape batch_size, seq_length, d_model
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        # Calculate the angles given the input
        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                 np.arange(d_model)[np.newaxis, :],
                                 d_model)
        # Calculate the positional encodings
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        # Expand the encodings with a new dimension
        pos_encoding = angles[np.newaxis, ...]

        return inputs + tf.cast(pos_encoding, tf.float32)


class ClsPositionEncodingLayer(layers.Layer):
    def __init__(self, d_model, dropout_rate, name="ClsPositionEncodingLayer"):
        super(ClsPositionEncodingLayer, self).__init__(name=name)

        self.cls_token_patch = tf.Variable(tf.random.normal((1, 1, d_model)))
        self.pos_embedding = PositionalEncoding()
        self.dropout_patch = layers.Dropout(dropout_rate)

    def call(self, inputs):
        cls_token_patch_tiled = tf.tile(
            self.cls_token_patch, [tf.shape(inputs)[0], 1, 1])

        outputs = tf.concat([cls_token_patch_tiled, inputs], axis=1)
        outputs = self.pos_embedding(outputs)
        outputs = self.dropout_patch(outputs)
        return outputs
