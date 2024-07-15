import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import time
import numpy as np
from utils.utils_mine import *
from utils.utils import *
from utils.utils_mine import generate_fnirs_adj_tf
from classifiers.layer.conv1d import conv1d_layer
from classifiers.layer.mamba import MambaBlock
from classifiers.layer.gnn import GNN
from classifiers.layer.rmsnorm import RMSNorm
from classifiers.model.jamba import Transformer_layer, Attention_MoE_layer, Mamba_layer, Mamba_MoE_layer
from classifiers.layer.mlp import MLP
class Classifier_Jamba():
    def __init__(self, output_directory, callbacks, input_shape, epochs, sweep_config, info):

        self.output_directory = output_directory
        self.callbacks = callbacks
        self.epochs = epochs

        self.info = info
        self.params = params = info['parameter']
        args = self.params['args']
        
        args.update_model_checkpoint(output_directory + 'checkpoint')
        self.callbacks.append(args.model_checkpoint)
        self.callbacks.append(args.earlystopping)
        self.callbacks.append(args.reduce_lr)                
        self.batch_size = args.batch_size

        # Define input shape
        inputs_time_point = tf.keras.Input(shape=input_shape[1:])
        x = inputs_time_point
        concate_input = []
        
        # MLP layersto do: should MLP be added here?
        if args.use_mlp_layer:
            x = layers.Dense(args.model_input_dims, activation=args.activation)(x)
            mlp_x = RMSNorm()(x)
            concate_input.append(mlp_x)
            x = mlp_x
        
        # Conv1D layer
        if args.use_conv1d_layer:
            x = conv1d_layer(args)(x)
            conv1d_x = RMSNorm()(x)
            concate_input.append(conv1d_x)
            x = conv1d_x
        
        # MambaBlock layer
        if args.use_mamba_block:
            x = MambaBlock(args)(tf.add_n(concate_input))
            mamba_x = RMSNorm()(x)
            concate_input.append(mamba_x)
            x = mamba_x
        
        # GNN layer
        if args.use_gnn_layer:
            adj = generate_fnirs_adj_tf()
            x = GNN(args.model_internal_dim, adj, args.activation, args.dropout_rate)(tf.add_n(concate_input))
            gnn_x = RMSNorm()(x)
            concate_input.append(gnn_x)
            x = gnn_x
        
        # Concatenate all the layers
        concat_x = tf.concat(concate_input, axis=-2)
        x = concat_x

        # Define outputs for multi-task learning
        output_list = []
        for metric_name, _ in args.metrics.items():
            task_x = x
            for _ in range(args.num_layers):
                # x = Mamba_layer(args)(x)
                # x = Mamba_MoE_layer(args)(x)   
                # x = Transformer_layer(args)(x)
                task_x = Attention_MoE_layer(args)(task_x)
                task_x = RMSNorm()(task_x)  # Ensure RMSNorm is defined or replace with LayerNormalization
            
            while task_x.shape[-1] > 64:
                task_x = layers.Dense(task_x.shape[-1] // 2, activation=args.activation)(task_x)
                task_x = RMSNorm()(task_x)  # Ensure RMSNorm is defined or replace with LayerNormalization
            
            if not args.use_lm_head:
                task_x = layers.Flatten()(task_x)
            task_x = layers.Dense(args.last_dense_units, activation=args.activation, kernel_regularizer=tf.keras.regularizers.l2(args.l2_rate))(task_x)
            task_x = layers.Dropout(args.dropout_rate)(task_x)
            task_x = RMSNorm()(task_x)  # Ensure RMSNorm is defined or replace with LayerNormalization

            output_metric = layers.Dense(args.num_classes, activation=args.final_activation, name=metric_name)(MLP(args)(task_x))
            output_list.append(output_metric)
            
        # Create model
        model = Model(inputs=inputs_time_point, outputs=output_list)
        model.summary()

        # Compile model
        model.compile(optimizer=args.optimizer,
                      loss=args.loss,
                      metrics=args.metrics)
        self.model = model

    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        start_time = time.time()
        if self.params['args'].load_previous_checkpoint == True:
            if os.path.exists(self.output_directory + 'checkpoint'):
                self.model.load_weights(self.output_directory + 'checkpoint')
        
        # y={'gender': Y_train[:, 0], 'age': Y_train[:, 1], 'education': Y_train[:, 2], 'smoking': Y_train[:, 3], 'alcohol': Y_train[:, 4], 'HAMD_Scores': Y_train[:, 5], 'Suicide_Risk': Y_train[:, 6], 'depression': Y_train[:, 7]},
        # validation_data=(X_val, {'gender': Y_val[:, 0], 'age': Y_val[:, 1], 'education': Y_val[:, 2], 'smoking': Y_val[:, 3], 'alcohol': Y_val[:, 4], 'HAMD_Scores': Y_val[:, 5], 'Suicide_Risk': Y_val[:, 6], 'depression': Y_val[:, 7]}),
        y = {metric_name: Y_train[:, i] for i, metric_name in enumerate(self.params['args'].metrics)}
        validation_y = {metric_name: Y_val[:, i] for i, metric_name in enumerate(self.params['args'].metrics)}

        hist = self.model.fit(
            x=X_train,
            y=y,
            validation_data=(X_val, validation_y),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=True,
            shuffle=True,
        )
        self.model.load_weights(self.output_directory + 'checkpoint')
            
        duration = time.time() - start_time
        self.info['duration'] = duration
        
        save_validation_acc_multi_task(self.output_directory, self.model.predict(X_val), Y_val, self.params['args'].metrics, self.info)
        save_validation_acc_multi_task(self.output_directory, self.model.predict(X_test), Y_test, self.params['args'].metrics, self.info, save_file_name='test_acc.txt')
        
        save_hist_file(hist, self.output_directory)
        # if check_if_save_model(self.output_directory, self.model.predict(X_test), Y_test, self.info['monitor_metric'], self.info):
        #     save_logs(self.model, self.output_directory, None, hist, Y_test_pred, Y_test, duration, lr=True, is_saving_checkpoint=False, hyperparameters=None)

        print(f'Training time is {duration}')
        if self.params.get('config_file_path') is not None:
            save_current_file_to_folder(self.params['config_file_path'] + [os.path.abspath(__file__)], self.output_directory)

    def predict(self):
        pass
