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

def run(target_shift):
    def get_taskwise_datasets(train_shift_idxs, test_shift_idxs):
        X_train, y_train = X[train_shift_idxs], y[train_shift_idxs]
        X_test, y_test = X[test_shift_idxs], y[test_shift_idxs]

        X_train_across_tasks, y_train_across_tasks = [], []
        X_test_across_tasks, y_test_across_tasks = [], []
        for task_idx in range(n_tasks - 1, -1, -1):
            train_idxs_of_task = np.where((y_train >= int(num_classes / n_tasks) * task_idx) & (y_train < int(num_classes / n_tasks) * (task_idx + 1)))[0]
            X_train_across_tasks.append(X_train[train_idxs_of_task])
            y_train_across_tasks.append(y_train[train_idxs_of_task] - int(num_classes / n_tasks) * task_idx)

            test_idxs_of_task = np.where((y_test >= int(num_classes / n_tasks) * task_idx) & (y_test < int(num_classes / n_tasks) * (task_idx + 1)))[0]
            X_test_across_tasks.append(X_test[test_idxs_of_task])
            y_test_across_tasks.append(y_test[test_idxs_of_task] - int(num_classes / n_tasks) * task_idx)

        return X_train_across_tasks, X_test_across_tasks, y_train_across_tasks, y_test_across_tasks

    def set_random_seed(random_seed):
        random.seed(random_seed)

    seeds = []
    shifts = []  
    tasks = []  
    org_accuracies = []  
    forward_accuracies = []

    forward_transfer_efficiencies_across_tasks_across_shifts = []
    def fill_in_forward_transfer_efficiencies_across_tasks(train_shift_idxs, test_shift_idxs, shift):
        X_train_across_tasks, X_test_across_tasks, y_train_across_tasks, y_test_across_tasks = get_taskwise_datasets(train_shift_idxs, test_shift_idxs)

        lifelong_dnn = LifeLongDNN()
        forward_transfer_efficiencies_across_tasks = []
        def fill_in_forward_transfer_efficiencies_per_task(task):
            X_train_of_task, y_train_of_task = X_train_across_tasks[task], y_train_across_tasks[task]
            X_test_of_task, y_test_of_task = X_test_across_tasks[task], y_test_across_tasks[task]

            set_random_seed(random_seed)
            lifelong_dnn.new_forest(X_train_of_task, y_train_of_task)

            forward_accuracy = np.mean(y_test_of_task == lifelong_dnn.predict(X_test_of_task, decider = task, representation = "all"))
            org_accuracy = np.mean(y_test_of_task == lifelong_dnn.predict(X_test_of_task, decider = task, representation = task))

            seeds.append(random_seed)
            shifts.append(shift)
            org_accuracies.append(org_accuracy)
            tasks.append(task)
            forward_accuracies.append(forward_accuracy)

            forward_transfer_efficiency_of_task = (1 - org_accuracy) / (1 - forward_accuracy)
            forward_transfer_efficiencies_across_tasks.append(forward_transfer_efficiency_of_task)

        for task in range(n_tasks):
            fill_in_forward_transfer_efficiencies_per_task(task)

            print("Forward Transfer Efficiencies: {}".format(forward_transfer_efficiencies_across_tasks))
        forward_transfer_efficiencies_across_tasks_across_shifts.append(forward_transfer_efficiencies_across_tasks)

    shift = 0
    for train_shift_idxs, test_shift_idxs in kfold.split(X, y):
        if shift == target_shift:
            fill_in_forward_transfer_efficiencies_across_tasks(train_shift_idxs, test_shift_idxs, shift)
        shift += 1
        
    forward_df = pd.DataFrame()
    forward_df['seed'] = seeds
    forward_df['shift'] = shifts
    forward_df['org_accuracy'] = org_accuracies
    forward_df['task'] = tasks
    forward_df['forward_accuracy'] = forward_accuracies

    pickle.dump(forward_df, open('../../pkls/FTE_shift_{}.p'.format(target_shift), 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--target_shift', type = int)
    parser.add_argument('--n_shifts', type = int)
    args = parser.parse_args()
    
    random_seed = 12345
    n_tasks = 10

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    y = y[:, 0]

    num_classes = len(np.unique(y))

    kfold = StratifiedKFold(n_splits = args.n_shifts, shuffle = False)
    
    run(args.target_shift)
                