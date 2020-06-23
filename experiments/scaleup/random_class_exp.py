#%%
import warnings
warnings.simplefilter("ignore")
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from itertools import product
import pandas as pd

import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil 

import sys
sys.path.append("../../src/")
from lifelong_dnn import LifeLongDNN
from joblib import Parallel, delayed
from multiprocessing import Pool

import tensorflow as tf

import time

from sklearn.model_selection import train_test_split

#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
#%%
def LF_experiment(data_x, data_y, ntrees, shift, slot, num_points_per_task, acorn=None):
    
    
    df = pd.DataFrame()
    shifts = []
    slots = []
    accuracies_across_tasks = []

    train_x_task0, train_y_task0, test_x_task0, test_y_task0 = cross_val_data(data_x, data_y, num_points_per_task, total_task=10, shift=shift, slot=slot)
    lifelong_forest = LifeLongDNN(model = "uf", parallel = True)
    lifelong_forest.new_forest(
            train_x_task0, 
            train_y_task0, 
            max_depth=ceil(log2(num_points_per_task)), 
            n_estimators=ntrees,
            parallel = True
            )
        
    task_0_predictions=lifelong_forest.predict(
        test_x_task0, representation='all', decider=0
        )

    shifts.append(shift)
    slots.append(slot)
    accuracies_across_tasks.append(np.mean(
        task_0_predictions == test_y_task0
        ))
    inference_times = []
    training_times = []
    
    for task_ii in range(29):
        train_x, train_y, _, _ = cross_val_data(data_x, data_y, num_points_per_task, total_task=10, shift=shift, slot=slot, task = task_ii)
        
        print("Starting Task {} For Fold {} For Slot {}".format(task_ii, shift, slot))
            
        start_time = time.time()
        lifelong_forest.new_forest(
            train_x, 
            train_y, 
            max_depth=ceil(log2(num_points_per_task)), 
            n_estimators=ntrees,
            parallel = False
            )
        intermediate_time = time.time()
        
        task_0_predictions=lifelong_forest.predict(
            test_x_task0, representation='all', decider=0
            )
        end_time = time.time()
            
        shifts.append(shift)
        slots.append(slot)
        accuracies_across_tasks.append(np.mean(
            task_0_predictions == test_y_task0
            ))
        print("Accuracies: {}".format(accuracies_across_tasks))
        training_times.append(intermediate_time - start_time)
        print("Training Times: {}".format(training_times))
        inference_times.append(end_time - intermediate_time)
        print("Inference Times: {}".format(inference_times))
            
    df['data_fold'] = shifts
    df['slot'] = slots
    df['accuracy'] = accuracies_across_tasks
    df['inference_time'] = inference_times
    df['training_time'] = training_times

    file_to_save = 'result/uf'+str(ntrees)+'_'+str(shift)+'_'+str(slot)+'.pickle'
    with open(file_to_save, 'wb') as f:
        pickle.dump(df, f)

#%%
def cross_val_data(data_x, data_y, num_points_per_task, total_task=10, shift=1, slot=0, task=0):
    skf = StratifiedKFold(n_splits=6)
    for _ in range(shift + 1):
        train_idx, test_idx = next(skf.split(data_x, data_y))
        
    data_x_train, data_y_train = data_x[train_idx], data_y[train_idx]
    data_x_test, data_y_test = data_x[test_idx], data_y[test_idx]
    
    selected_classes = np.random.choice(range(0, 100), 10)
    train_idxs_of_selected_class = np.array([np.where(data_y_train == y_val)[0] for y_val in selected_classes])
    num_points_per_class_per_slot = [int(len(train_idxs_of_selected_class[class_idx]) // 10) for class_idx in range(len(selected_classes))]
    selected_idxs = np.concatenate([np.random.choice(train_idxs_of_selected_class[class_idx], num_points_per_class_per_slot[class_idx]) for class_idx in range(len(selected_classes))])
    train_idxs = np.random.choice(selected_idxs, num_points_per_task)
    data_x_train = data_x_train[train_idxs]
    data_y_train = data_y_train[train_idxs]
    
    test_idxs_of_selected_class = np.concatenate([np.where(data_y_test == y_val)[0] for y_val in selected_classes])
    data_x_test = data_x_test[test_idxs_of_selected_class]
    data_y_test = data_y_test[test_idxs_of_selected_class]
    
    
    return data_x_train, data_y_train, data_x_test, data_y_test

#%%
def run_parallel_exp(data_x, data_y, n_trees, num_points_per_task, slot=0, shift=1):
    LF_experiment(data_x, data_y, n_trees, shift, slot, num_points_per_task, acorn=12345)

#%%
### MAIN HYPERPARAMS ###
num_points_per_task = 500
########################

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
data_x = data_x.reshape((data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3]))
data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]


#%%
slot_fold = range(0, 10)
shift_fold = range(1,7,1)
n_trees=[10]
iterable = product(n_trees,shift_fold, slot_fold)
for ntree,shift,slot in iterable:
    run_parallel_exp(data_x, data_y, ntree, num_points_per_task, slot=slot, shift=shift) 
            

# %%
