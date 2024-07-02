import numpy as np
import pandas as pd
import os
import time
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, recall_score
import random
from utils.utils_mine import *
from utils.utils import *
# Assuming other necessary imports and utility functions are defined elsewhere as before

class Classifier_DecisionTree():
    def __init__(self, output_directory, callbacks, input_shape, epochs, sweep_config, info):
        self.output_directory = output_directory
        self.info = info
        
        # Decision tree does not use epochs, but other hyperparameters can be randomized or fixed
        params = {
            'criterion': random.choice(['gini', 'entropy']),  # Function to measure the quality of a split
            'max_depth': random.choice([3, 4, 5, 6, 7, 8, 9, 10]),  # Maximum depth of the tree
            'min_samples_split': random.choice([2, 3, 4]),  # Minimum number of samples required to split an internal node
            'min_samples_leaf': random.choice([1, 2, 3]),  # Minimum number of samples required to be at a leaf node
        }
        self.params = params
        self.model = DecisionTreeClassifier(
            criterion=params['criterion'], 
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf']
        )

        self.hyperparameters = params
        print(f'hyperparameters: {self.hyperparameters}')

    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        start_time = time.time()
        
        # Fit the model
        model_path = self.output_directory + 'checkpoint'
        if os.path.exists(model_path):
            self.model.load_weights(model_path)        
        
        hist = self.model.fit(X_train, Y_train)

        Y_pred = self.model.predict(X_test)
        self.info['Y_pred_in_test'] = Y_pred
        Y_test_pred = self.model.predict(X_test)
        Y_true = np.argmax(Y_test, axis=1)
        
        Y_val_pred = np.argmax(self.model.predict(X_val), axis=1)
        Y_val_true = np.argmax(Y_val, axis=1)

        duration = time.time() - start_time
        self.info['duration'] = duration
        save_validation_acc(self.output_directory, self.model.predict(X_val), Y_val, self.info['monitor_metric'], self.info)
        save_validation_acc(self.output_directory, self.model.predict(X_test), Y_test, self.info['monitor_metric'], self.info,
                            save_file_name='test_acc.txt')
        if check_if_save_model(self.output_directory, self.model.predict(X_test), Y_test, self.info['monitor_metric'], self.info):
            # save learning rate as well
            # Can ignore the result name which has beend set as None
            save_logs(self.model, self.output_directory, None,
                      hist, Y_pred, Y_true, duration,
                      lr=True,
                      is_saving_checkpoint=False,
                      hyperparameters=self.hyperparameters,
                      y_true_onehot=Y_test,
                      y_pred_onehot=tf.one_hot(Y_pred, depth=2).numpy(),
                      pass_history=True
                      )

    def predict(self):
        pass
        # Implement prediction logic if necessary

