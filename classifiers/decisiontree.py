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
        hist = self.model.fit(X_train, Y_train)

        # Predictions
        Y_pred = self.model.predict(X_test)
        # For decision trees, Y_pred is already the predicted classes; no need for np.argmax
        Y_true = Y_test  # Assuming Y_test is already in the correct format

        duration = time.time() - start_time
        # Save logs, handle validations, etc., similarly as before but adjusted for Decision Tree specifics
        Y_pred = np.argmax(Y_pred, axis=1)
        Y_true = np.argmax(Y_test, axis=1)

        save_validation_acc(self.output_directory, np.argmax(
            self.model.predict(X_val), axis=1), np.argmax(Y_val, axis=1), self.info['monitor_metric'], self.info)
        if check_if_save_model(self.output_directory, Y_pred, Y_true, self.info['monitor_metric'], self.info):
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
