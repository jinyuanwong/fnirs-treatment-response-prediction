#residual gated graph convolutional neural network
from classifiers.layer.RGGCN import ResGatedGCNLayer
import tensorflow as tf

class ResGatedGCNModel(tf.keras.Model):
    def __init__(self, units, num_layers, **kwargs):
        super(ResGatedGCNModel, self).__init__(**kwargs)
        self.layers_list = [ResGatedGCNLayer(units) for _ in range(num_layers)]

    def call(self, inputs, adjacency_matrix):
        x = inputs
        for index, layer in enumerate(self.layers_list):
            x = layer(x, adjacency_matrix) + x
        return x
    
