
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, Model

class GNN(tf.keras.Model):
    def __init__(self,
                 d_model,
                 adj,
                 activation,
                 dropout_rate):
        super(GNN, self).__init__()

        self.W = layers.Dense(units=d_model, activation=activation)
        self.dropout = layers.Dropout(dropout_rate)
        self.adj = self.normalize_adjacency(adj)
        # self.activation = activation
        # self.dropout_rate = dropout_rate

    def normalize_adjacency(self, adj):
        d = tf.reduce_sum(adj, axis=-1)
        d_sqrt_inv = tf.pow(d, -0.5)
        d_sqrt_inv = tf.where(tf.math.is_inf(d_sqrt_inv), 0., d_sqrt_inv)
        d_mat_inv_sqrt = tf.linalg.diag(d_sqrt_inv)
        return tf.matmul(tf.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    def call(self, inputs):
        # adj_normalized = self.normalize_adjacency(adj)
        # inputs_features = self.W(inputs)
        # outputs = tf.linalg.matmul(adj_normalized, inputs_features)
        # adj_normalized = self.normalize_adjacency(adj)
        # inputs_features = self.W(inputs)
        outputs = tf.linalg.matmul(self.adj, inputs)
        outputs = self.W(outputs)
        outputs = self.dropout(outputs)
        return outputs
