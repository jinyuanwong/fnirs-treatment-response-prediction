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
from classifiers.layer.gnn import gnn_layer
from classifiers.layer.rmsnorm import RMSNorm
from classifiers.model.jamba import Transformer_layer, Attention_MoE_layer, Mamba_layer, Mamba_MoE_layer
from classifiers.layer.mlp import MLP
from utils.fnirs_utils import delete_checkpoints
import json
class Classifier_Jamba():
    def __init__(self, output_directory, callbacks, input_shape, epochs, sweep_config, info):

        self.output_directory = output_directory
        self.callbacks = callbacks
        self.epochs = epochs

        self.info = info
        self.params = params = info['parameter']
        args = self.params['args']
        self.class_weights_dict = {0: 1, 1: args.classweight1}
        args.update_model_checkpoint(output_directory + 'checkpoint')
        self.callbacks.append(args.model_checkpoint)
        self.callbacks.append(args.earlystopping)
        if args.reduce_lr is not None: self.callbacks.append(args.reduce_lr)                
        self.batch_size = args.batch_size
        # Define input shape
        inputs_time_point = tf.keras.Input(shape=input_shape[1:])
        x = inputs_time_point
        

        
        # Define layers
        if args.use_conv1d_layer:        
            x = conv1d_layer(args)(x)
            x = args.normalization_layer()(x)
        # move mlp_layer from "before conv1d" to "after conv1d"
        if args.use_mlp_layer:                
            x = layers.Dense(args.model_input_dims, activation=args.activation)(x)
            x = args.normalization_layer()(x)

        if args.use_mamba_block: 
            x = MambaBlock(args)(x)
            x = args.normalization_layer()(x)
        # move the gnn layer to first 
        
        if args.use_gnn_layer: 
            adj = generate_fnirs_adj_tf()
            x = gnn_layer(args.gnn_type)(args.model_internal_dim, adj, args.activation, args.dropout_rate)(x)
            x = args.normalization_layer()(x)
            
        for _ in range(args.num_layers):
            # x = Mamba_layer(args)(x)
            # x = Mamba_MoE_layer(args)(x)   
            x = Transformer_layer(args)(x)
            # x = Attention_MoE_layer(args)(x)
            x = args.normalization_layer()(x)
        # x = tf.concat([x, conv1d_layer(args)(x)], axis=-1)
        # x = layers.GlobalAveragePooling1D(data_format='channels_first')(x)

        if not args.use_lm_head: 
            x = layers.Flatten()(x)

 
        
        # x = layers.Dense(args.last_dense_units * 4, activation=args.activation, kernel_regularizer=tf.keras.regularizers.l2(args.l2_rate))(x)
        # x = layers.Dropout(args.dropout_rate)(x)
        x = layers.Dense(args.last_dense_units, activation=args.activation, kernel_regularizer=tf.keras.regularizers.l2(args.l2_rate))(x)
        x = layers.Dropout(args.dropout_rate)(x)
        x = args.normalization_layer()(x)

        if args.branch_revert_last_two_dimension:
            y = tf.transpose(inputs_time_point, perm=[0, 2, 1])
            y = layers.Dense(args.model_input_dims, activation=args.activation)(y)
            y = layers.GlobalAveragePooling1D(data_format='channels_first')(y)#
            y = args.normalization_layer()(y)
            y = layers.Flatten()(y)
            x = tf.concat([x, y], axis=-1)       
                    
        output_list = []
        # Define outputs
        for metric_name, _ in  args.metrics.items():
            output_metric = layers.Dense(args.num_classes, activation=args.final_activation, name=metric_name)(MLP(args)(x))
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
            # class_weight=self.class_weights_dict
        )
        self.model.load_weights(self.output_directory + 'checkpoint')
        
            
        Y_test_pred = self.model.predict(X_test)
        Y_val_pred = self.model.predict(X_val)

        duration = time.time() - start_time
        self.info['duration'] = duration
        
        val_performance_metrics = save_validation_acc_multi_task(self.output_directory, self.model.predict(X_val), Y_val, self.params['args'].metrics, self.info)
        test_performance_metrics = save_validation_acc_multi_task(self.output_directory, self.model.predict(X_test), Y_test, self.params['args'].metrics, self.info, save_file_name='test_acc.txt')
        
        # save_hist_file(hist, self.output_directory)
        # if check_if_save_model(self.output_directory, self.model.predict(X_test), Y_test, self.info['monitor_metric'], self.info):
        #     save_logs(self.model, self.output_directory, None, hist, Y_test_pred, Y_test, duration, lr=True, is_saving_checkpoint=False, hyperparameters=None)

        # print(f'Training time is {duration}')
        # if self.params.get('config_file_path') is not None:
        #     save_current_file_to_folder(self.params['config_file_path'] + [os.path.abspath(__file__)], self.output_directory)

        # SQL operation
        serialized_history = {key: [float(val) for val in values] for key, values in hist.history.items()}
        serialized_val_performance_metrics = {key: [float(val) for val in values] for key, values in val_performance_metrics.items()}
        serialized_test_performance_metrics = {key: [float(val) for val in values] for key, values in test_performance_metrics.items()}
        performance_output = {
            'history': json.dumps(serialized_history),
            'y_test_true': json.dumps(Y_test.tolist()),
            'y_test_pred':json.dumps(Y_test_pred.tolist()),
            'y_val_true': json.dumps(Y_val.tolist()),
            'y_val_pred': json.dumps(Y_val_pred.tolist()),
            'duration': duration,
            'name_performance_metrics': ','.join(val_performance_metrics.columns.to_list()),
            'val_performance_metrics': json.dumps(serialized_val_performance_metrics),
            'test_performance_metrics': json.dumps(serialized_test_performance_metrics),
        }
        
        # delete checkpoint if self.params['args'].delete_checkpoint == True
        if self.params['args'].delete_checkpoint:
            delete_checkpoints(self.output_directory)
        return performance_output

    def predict(self):
        pass
