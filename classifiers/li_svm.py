import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time

import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.optimizers as ko
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils.utils_mine import *
from utils.utils import *
import tensorflow.keras as keras
from tensorflow.keras.initializers import HeNormal
from sklearn.metrics import auc, accuracy_score, roc_curve, recall_score
import tensorflow_addons as tfa
import random
import os
import pandas as pd
import math
from xgboost import XGBClassifier
import wandb
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import SVC
from deap import base, creator, tools
from sklearn.model_selection import cross_val_score, train_test_split


class Classifier_LI_SVM():
    def __init__(self, output_directory, callbacks, input_shape, epochs, sweep_config, info):
        # input_shape = (200, 52, 128, 1)

        self.output_directory = output_directory
        self.info = info
        # 随机给定超参数进行训练
        # 32#random.choice([16, 32, 48])  # 128 256
        params = {}
        self.params = params
        self.model = SVC(probability=True)
        
        self.hyperparameters = params
        print(f'hyperparameters: {self.hyperparameters}')
    def genetic_algorithm(self, data, label):
        X_train, X_test, Y_train, Y_test = train_test_split(
        data, label, test_size=0.25)
        
        train_best_acc_detail = []
        test_best_acc_detail = []
        # Total features
        n_features = 520  # 52 channels each with 10 features

        # Number of populations
        n_population = 200

        # Number of generations
        n_generation = 300

        # Probability with which two individuals are crossed
        cxpb = 0.5
        # Probability for mutating an individual
        mutpb = 0.2

        # If you want the GA to prioritize one over the other, you'll need to adjust the weights when you create the FitnessMax. For example, if you only care about test accuracy, you can set the weight of the training accuracy to 0:
        # creator.create("FitnessMax", base.Fitness, weights=(0.0, 1.0))
        creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", np.random.randint, 0, 2)
        toolbox.register("individual", tools.initRepeat,
                        creator.Individual, toolbox.attr_bool, n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def eval_features(individual):
            # Select features where bit is 1
            selected_features = [i for i, bit in enumerate(individual) if bit == 1]
            # If no features selected, return a very poor fitness
            if len(selected_features) == 0:
                return (0.0,)
            # Extract these features from the dataset
            X_selected_train = X_train[:, selected_features]
            # Train and evaluate a classifier
            clf = SVC()
            # , test = ....
            scores = cross_val_score(clf, X_selected_train, Y_train, cv=5)

            # Train the classifier on the entire training dataset
            clf.fit(X_selected_train, Y_train)

            # Evaluate the classifier on the test dataset
            X_selected_test = X_test[:, selected_features]
            test_score = clf.score(X_selected_test, Y_test)

            return (scores.mean(), test_score)

        toolbox.register("evaluate", eval_features)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Now you can initialize population and run the algorithm.
        # Create an initial population of 200 individuals (each individual is a combination of features)
        population = toolbox.population(n=n_population)

        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        all_gen_fit_result = []
        for gen in range(n_generation):
            train_best_acc_detail.append(np.max([ind.fitness.values[0] for ind in population]))
            test_best_acc_detail.append(np.max([ind.fitness.values[1] for ind in population]))
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                all_gen_fit_result.append(fit)
            # Replace the old population by the offspring
            population[:] = offspring

        # Extract and print the final statistics
        # fits = [ind.fitness.values[0] for ind in population]
        # length = len(population)
        # mean = sum(fits) / length
        # sum2 = sum(x*x for x in fits)
        # std = abs(sum2 / length - mean**2)**0.5

        # Select 1 individual from the population
        best_ind = tools.selBest(population, 1)[0]
        # selected_features = [index for index,
        #                     bit in enumerate(best_ind) if bit == 1]        
        best_ind = np.array(best_ind)
        selected_features = best_ind == 1
        return selected_features
        
    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):


        Y_train, Y_val, Y_test = np.argmax(Y_train, axis=1), np.argmax(Y_val, axis=1), np.argmax(Y_test, axis=1)
        start_time = time.time()
        selected_features = self.genetic_algorithm(X_train, Y_train)
        X_train = X_train[:, selected_features]
        X_test = X_test[:, selected_features]
        X_val = X_val[:, selected_features]
        
        hist = self.model.fit(X_train, Y_train)

        Y_pred = self.model.predict(X_test)
        # Y_pred = np.argmax(Y_pred, axis=1)
        Y_true = Y_test #np.argmax(Y_test, axis=1)

        duration = time.time() - start_time
        self.info['duration'] = duration
        save_validation_acc(self.output_directory, self.model.predict(X_val), Y_val, self.info['monitor_metric'], self.info)
        if check_if_save_model(self.output_directory, Y_pred, Y_true, self.info['monitor_metric'], self.info):
            # save learning rate as well
            # Can ignore the result name which has beend set as None
            save_logs(self.model, self.output_directory, None,
                      hist, Y_pred, Y_true, duration,
                      lr=True,
                      is_saving_checkpoint=False,
                      hyperparameters=self.hyperparameters,
                      y_true_onehot=tf.one_hot(Y_test, depth=2).numpy(),
                      y_pred_onehot=tf.one_hot(Y_pred, depth=2).numpy(),
                      pass_history=True
                      )

        print(f'Training time is {duration}')
        save_current_file_to_folder(os.path.abspath(__file__), self.output_directory)
        if self.params.get('config_file_path') is not None:
            save_current_file_to_folder(self.params['config_file_path'], self.output_directory)

    def predict(self):
        pass
