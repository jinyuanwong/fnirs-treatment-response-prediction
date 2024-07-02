import tensorflow as tf 
from tensorflow.keras import layers 
from tensorflow import keras

class MoE(layers.Layer):
    def __init__(self, 
                 args,
                 **kwargs):
        super(MoE, self).__init__(**kwargs)
        self.n_experts = args.n_experts
        self.d_experts = args.model_internal_dim
        self.activation = args.activation
        self.experts = [layers.Dense(args.model_internal_dim, activation=args.activation, kernel_regularizer=keras.regularizers.l2(args.l2_rate)) for _ in range(args.n_experts)]
        self.gating = layers.Dense(args.n_experts, activation='softmax')
    
    def call(self, inputs):
        expert_outputs = [expert(inputs) for expert in self.experts]
        expert_outputs = tf.stack(expert_outputs, axis=-1)
        gate_outputs = self.gating(inputs)
        weight_sum = tf.einsum('bln,bldn->bld', gate_outputs, expert_outputs)
        return weight_sum        
        