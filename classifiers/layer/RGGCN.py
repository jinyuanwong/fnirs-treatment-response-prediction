#residual gated graph convolutional neural network
import tensorflow as tf

class ResGatedGCNLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(ResGatedGCNLayer, self).__init__(**kwargs)
        self.units = units
        self.U = self.add_weight(shape=(units, units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='U')
        self.V = self.add_weight(shape=(units, units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='V')
        self.A = self.add_weight(shape=(units, units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='A')
        self.B = self.add_weight(shape=(units, units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='B')
        self.bias = self.add_weight(shape=(units,),
                                    initializer='zeros',
                                    trainable=True,
                                    name='bias')

    def call(self, inputs, adjacency_matrix):
        # inputs: Node features (batch_size, num_nodes, num_features)
        # adjacency_matrix: Adjacency matrix (num_nodes, num_nodes)
        
        # Graph convolution with U weight matrix
        conv_output = tf.matmul(adjacency_matrix, inputs)
        conv_output = tf.matmul(conv_output, self.U)
        
        # Edge gating mechanism with A and B weight matrices
        edge_gates = tf.sigmoid(tf.matmul(inputs, self.A) + tf.matmul(inputs, self.B))
        
        # Gated graph convolution with V weight matrix
        gated_conv_output = tf.matmul(adjacency_matrix, inputs)
        gated_conv_output = tf.matmul(gated_conv_output, self.V)
        gated_conv_output = tf.multiply(gated_conv_output, edge_gates)
        
        # Residual connection and bias
        output = gated_conv_output + inputs
        output = tf.nn.relu(output + self.bias)
        
        return output