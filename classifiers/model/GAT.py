#residual gated graph convolutional neural network
from classifiers.layer.GAT import GraphAttention
import tensorflow as tf

class GAT(tf.keras.Model):
    def __init__(self, units, num_layers, **kwargs):
        super(GAT, self).__init__(**kwargs)
        self.layers_list = [GraphAttention(units) for _ in range(num_layers)]

    def call(self, inputs, adjacency_matrix):
        x = inputs
        for index, layer in enumerate(self.layers_list):
            x = layer(x, adjacency_matrix)
        return x
    
