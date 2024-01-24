import tensorflow as tf
from tensorflow.keras import layers

class GINLayer(layers.Layer):
    def __init__(self, hidden_dim, eps_initializer='zeros'):
        super(GINLayer, self).__init__()
        self.linear_1 = layers.Dense(hidden_dim)
        self.linear_2 = layers.Dense(hidden_dim)
        self.linear_3 = layers.Dense(hidden_dim)
        self.batch_norm = layers.BatchNormalization()
        self.eps = tf.Variable(initial_value=tf.keras.initializers.get(eps_initializer)(shape=(1,)),
                               trainable=True, name='epsilon')

    def call(self, inputs, adjacency_matrix):
        neighbor_sum = tf.matmul(adjacency_matrix, inputs)
        self_features = inputs + self.eps * inputs  # Add the (1 + epsilon) * self term
        output = self.linear_1(neighbor_sum + self_features)
        output = self.linear_2(output)
        output = self.linear_3(output)
        output = self.batch_norm(output)
        output = tf.nn.relu(output)
        return output


class GINModel(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(GINModel, self).__init__()
        self.gin_layer1 = GINLayer(hidden_dim)
        self.gin_layer2 = GINLayer(hidden_dim)
        # self.global_pool = layers.GlobalAveragePooling1D()
        # self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, adjacency_matrix):
        x = self.gin_layer1(inputs, adjacency_matrix)
        x = self.gin_layer2(x, adjacency_matrix)
        # x = self.global_pool(x)
        # output = self.classifier(x)
        return x
