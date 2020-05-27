from math import ceil 

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
            y_train_across_tasks.append(y_train[train_idxs_of_task])

            test_idxs_of_task = np.where((y_test >= int(num_classes / n_tasks) * task_idx) & (y_test < int(num_classes / n_tasks) * (task_idx + 1)))[0]
            X_test_across_tasks.append(X_test[test_idxs_of_task])
            y_test_across_tasks.append(y_test[test_idxs_of_task])

        return X_train_across_tasks, X_test_across_tasks, y_train_across_tasks, y_test_across_tasks        

    backward_transfer_efficiencies_across_tasks_across_shifts = [[] for _ in range(n_tasks)]
    def fill_in_transfer_efficiencies_across_tasks(train_shift_idxs, test_shift_idxs, shift):
        
        X_train_across_tasks, X_test_across_tasks, y_train_across_tasks, y_test_across_tasks = get_taskwise_datasets(train_shift_idxs, test_shift_idxs)
        
        single_task_accuracies = []
        
        lifelong_dnn = LifeLongDNN(model = "uf")
        for task in range(n_tasks):
            print("Adding Forest For Task: {}".format(task))
            X_train_of_task = X_train_across_tasks[task]
            y_train_of_task = y_train_across_tasks[task]
            np.random.seed(random_seed)
            lifelong_dnn.new_forest(X_train_of_task , y_train_of_task, n_estimators = 10, max_depth = ceil(np.log2(len(X_train_of_task))))
            single_task_accuracy = np.mean(y_test_across_tasks[task] == lifelong_dnn.predict(X_test_across_tasks[task], decider = task, representation = task))
            single_task_accuracies.append(single_task_accuracy)
            
        df_single_task = pd.DataFrame()
        df_single_task['task'] = range(1, 11)
        df_single_task['data_fold'] = shift
        df_single_task['accuracy'] = single_task_accuracies
        print(df_single_task)

        

        def fill_in_transfer_efficiencies_per_task(base_task):
            shifts = []
            tasks = []
            accuracies_across_tasks = []
            errors = []
            for task in range(base_task + 1):
                accuracy = np.mean(y_test_across_tasks[task] == lifelong_dnn.predict(X_test_across_tasks[task], decider = task, representation = range(base_task + 1)))
                accuracies_across_tasks.append(accuracy)
                
                shifts.append(shift)
                tasks.append(task + 1)
            print("Accuracies of With Base Task {} Across Tasks: {}".format(base_task + 1, accuracies_across_tasks))
                
            df = pd.DataFrame()
            df['data_fold'] = shifts
            df['task'] = tasks
            df['base_task'] = base_task + 1
            df['accuracy'] = accuracies_across_tasks

            return df

        
        df_across_tasks = Parallel(n_jobs=-1)(delayed(fill_in_transfer_efficiencies_per_task)(task) for task in range(n_tasks))
        
        return pd.concat(df_across_tasks, ignore_index = True), df_single_task

    shift = 0
    for train_shift_idxs, test_shift_idxs in kfold.split(X, y):
        if shift == target_shift:
            df_tuple = fill_in_transfer_efficiencies_across_tasks(train_shift_idxs, test_shift_idxs, shift)
            pickle.dump(df_tuple, open('pkls/shift_{}.p'.format(target_shift), 'wb'))
        shift += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--target_shift', type = int)
    parser.add_argument('--n_shifts', type = int)
    args = parser.parse_args()
    
    n_tasks = 10
    random_seed = 1234

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
    X = np.concatenate([X_train, X_test])
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
    y = np.concatenate([y_train, y_test])
    y = y[:, 0]

    num_classes = len(np.unique(y))

    kfold = StratifiedKFold(n_splits = args.n_shifts, shuffle = False)
    
    run(args.target_shift)
                