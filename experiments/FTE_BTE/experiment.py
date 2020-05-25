import random

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold

import sys
sys.path.append("../../src/")
from lifelong_dnn import LifeLongDNN

import pandas as pd

import argparse

from joblib import Parallel, delayed

def run(target_shift):
    def get_taskwise_datasets(train_shift_idxs, test_shift_idxs):
        X_train, y_train = X[train_shift_idxs], y[train_shift_idxs]
        X_test, y_test = X[test_shift_idxs], y[test_shift_idxs]

        X_train_across_tasks, y_train_across_tasks = [], []
        X_test_across_tasks, y_test_across_tasks = [], []
        for task_idx in range(n_tasks):
            train_idxs_of_task = np.where((y_train >= int(num_classes / n_tasks) * task_idx) & (y_train < int(num_classes / n_tasks) * (task_idx + 1)))[0]
            X_train_across_tasks.append(X_train[train_idxs_of_task])
            y_train_across_tasks.append(y_train[train_idxs_of_task] - int(num_classes / n_tasks) * task_idx)

            test_idxs_of_task = np.where((y_test >= int(num_classes / n_tasks) * task_idx) & (y_test < int(num_classes / n_tasks) * (task_idx + 1)))[0]
            X_test_across_tasks.append(X_test[test_idxs_of_task])
            y_test_across_tasks.append(y_test[test_idxs_of_task] - int(num_classes / n_tasks) * task_idx)

        return X_train_across_tasks, X_test_across_tasks, y_train_across_tasks, y_test_across_tasks        

    backward_transfer_efficiencies_across_tasks_across_shifts = [[] for _ in range(n_tasks)]
    def fill_in_transfer_efficiencies_across_tasks(train_shift_idxs, test_shift_idxs, shift):
        
        X_train_across_tasks, X_test_across_tasks, y_train_across_tasks, y_test_across_tasks = get_taskwise_datasets(train_shift_idxs, test_shift_idxs)
        
        lifelong_dnn = LifeLongDNN(model = "uf")
        for task in range(n_tasks):
            print("Adding Forest For Task: {}".format(task))
            random.seed(random_seed)
            X_train_of_task = X_train_across_tasks[task]
            y_train_of_task = y_train_across_tasks[task]
            lifelong_dnn.new_forest(X_train_of_task , y_train_of_task, n_estimators = 41, max_depth = np.log2(len(X_train_of_task)))
        
        

        def fill_in_transfer_efficiencies_per_task(task):
            seeds = []
            shifts = []
            tasks = []
            tasks_seen = []
            reverse_accuracies = []
            forward_accuracies = []
            
            backward_accuracies_across_tasks = []
            forward_accuracies_across_tasks = []

            def fill_in_accuracies_per_task(task_seen):
                backward_accuracy = np.mean(y_test_across_tasks[task] == lifelong_dnn.predict(X_test_across_tasks[task], decider = task, representation = range(task, task_seen + 1)))
                backward_accuracies_across_tasks.append(backward_accuracy)
                
                forward_accuracy = np.mean(y_test_across_tasks[task_seen] == lifelong_dnn.predict(X_test_across_tasks[task_seen], decider = task_seen, representation = range(task, task_seen + 1)))
                forward_accuracies_across_tasks.append(forward_accuracy)

            for task_seen in range(task, n_tasks):
                fill_in_accuracies_per_task(task_seen)
                print("Backward Accuracies of Task {} Across Tasks: {}".format(task + 1, backward_accuracies_across_tasks))
                print("Forward Accuracies of Task {} Across Tasks: {}".format(task + 1, forward_accuracies_across_tasks))
                
                seeds.append(random_seed)
                shifts.append(shift)
                tasks.append(task)
                tasks_seen.append(task_seen)
                reverse_accuracies.append(backward_accuracies_across_tasks[-1])
                forward_accuracies.append(forward_accuracies_across_tasks[-1])
                
            df = pd.DataFrame()
            df['seed'] = seeds
            df['shift'] = shifts
            df['task'] = tasks
            df['task_seen'] = tasks_seen
            df['reverse_accuracy'] = reverse_accuracies
            df['forward_accuracy'] = forward_accuracies

            return df

        
        df_across_tasks = Parallel(n_jobs=-1)(delayed(fill_in_transfer_efficiencies_per_task)(task) for task in range(n_tasks))
        return pd.concat(df_across_tasks, ignore_index = True)

    shift = 0
    for train_shift_idxs, test_shift_idxs in kfold.split(X, y):
        if shift == target_shift:
            df = fill_in_transfer_efficiencies_across_tasks(train_shift_idxs, test_shift_idxs, shift)
            pickle.dump(df, open('pkls/shift_{}.p'.format(target_shift), 'wb'))
        shift += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--target_shift', type = int)
    parser.add_argument('--n_shifts', type = int)
    args = parser.parse_args()
    
    random_seed = 12345
    n_tasks = 10

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
    X = np.concatenate([X_train, X_test])
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
    y = np.concatenate([y_train, y_test])
    y = y[:, 0]

    num_classes = len(np.unique(y))

    kfold = StratifiedKFold(n_splits = args.n_shifts, shuffle = False)
    
    run(args.target_shift)
                