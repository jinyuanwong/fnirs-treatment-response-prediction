import tensorflow as tf 
from tensorflow.keras import layers 

class MoE(layers.Layer):
    def __init__(self, 
                 n_experts, 
                 d_experts, 
                 activation,
                 **kwargs):
        super(MoE, self).__init__(**kwargs)
        self.n_experts = n_experts
        self.d_experts = d_experts
        self.activation = activation
        self.experts = [layers.Dense(d_experts, activation=activation) for _ in range(n_experts)]
        self.gating = layers.Dense(n_experts, activation='softmax')
    
    def call(self, inputs):
        expert_outputs = [expert(inputs) for expert in self.experts]
        expert_outputs = tf.stack(expert_outputs, axis=-1)
        gate_outputs = self.gating(inputs)
        weight_sum = tf.einsum('bln,bldn->bld', gate_outputs, expert_outputs)
        return weight_sum        
        