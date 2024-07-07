import tensorflow as tf 
from tensorflow.keras import layers 
from tensorflow import keras

class MLP(layers.Layer):
    def __init__(self, 
                 args,
                 **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.mlp1 = layers.Dense(args.model_internal_dim, activation=args.activation, kernel_regularizer=keras.regularizers.l2(args.l2_rate))
        self.dropout1 = layers.Dropout(args.dropout_rate)
        self.mlp2 = layers.Dense(args.model_internal_dim//2, activation=args.activation, kernel_regularizer=keras.regularizers.l2(args.l2_rate))
        self.dropout2 = layers.Dropout(args.dropout_rate)
        self.mlp3 = layers.Dense(args.model_internal_dim//4, activation=args.activation, kernel_regularizer=keras.regularizers.l2(args.l2_rate))
        # self.dropout3 = layers.Dropout(args.dropout_rate)

        
    
    def call(self, inputs):
        output = self.mlp1(inputs)
        output = self.dropout1(output)
        output = self.mlp2(output)
        output = self.dropout2(output)
        output = self.mlp3(output)
        return output
        