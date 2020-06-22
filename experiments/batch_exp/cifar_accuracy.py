#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from itertools import product
import pandas as pd

import numpy as np
import pickle
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from math import log2, ceil 

import sys
sys.path.append("../../src/")
from lifelong_dnn import LifeLongDNN
from joblib import Parallel, delayed
from multiprocessing import Pool

import tensorflow as tf

# %%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# %%
def LF_experiment(train_x, train_y, test_x, test_y, ntrees, shift, slot, num_points_per_task, acorn=None):
    
    RF = BaggingClassifier(
                DecisionTreeClassifier(
                max_depth=ceil(log2(num_points_per_task*10)),
                min_samples_leaf=1,
                max_features="auto"
            ),
            n_estimators=ntrees,
            max_samples=0.63,
            n_jobs = -1
        )
    
    if acorn is not None:
        np.random.seed(acorn)

    RF.fit(
            train_x, 
            train_y
            )

    llf_task=RF.predict(
            test_x
            )

    rf_accuracy = np.mean(
            llf_task == test_y
            )

    print(rf_accuracy)
      
    return rf_accuracy


# %%
def cross_val_data(data_x, data_y, num_points_per_task, total_task=10, shift=1):
    x = data_x.copy()
    y = data_y.copy()
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]
    
    batch_per_task=5000//num_points_per_task
    sample_per_class = num_points_per_task//total_task
    test_data_slot=100//batch_per_task

    for task in range(total_task):
        for batch in range(batch_per_task):
            for class_no in range(task*10,(task+1)*10,1):
                indx = np.roll(idx[class_no],(shift-1)*100)
                
                if batch==0 and class_no==0 and task==0:
                    train_x = x[indx[batch*sample_per_class:(batch+1)*sample_per_class],:]
                    train_y = y[indx[batch*sample_per_class:(batch+1)*sample_per_class]]
                    test_x = x[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500],:]
                    test_y = y[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500]]
                else:
                    train_x = np.concatenate((train_x, x[indx[batch*sample_per_class:(batch+1)*sample_per_class],:]), axis=0)
                    train_y = np.concatenate((train_y, y[indx[batch*sample_per_class:(batch+1)*sample_per_class]]), axis=0)
                    test_x = np.concatenate((test_x, x[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500],:]), axis=0)
                    test_y = np.concatenate((test_y, y[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500]]), axis=0)
            
    return train_x, train_y, test_x, test_y


# %%
def run_parallel_exp(data_x, data_y, n_trees, num_points_per_task, slot=0, shift=1):
    train_x, train_y, test_x, test_y = cross_val_data(data_x, data_y, num_points_per_task, shift=shift)
    
    rf_accuracy = LF_experiment(train_x, train_y, test_x, test_y, n_trees, shift, slot, num_points_per_task, acorn=12345)

    return rf_accuracy

#%%
### MAIN H≈YPERPARAMS ###
model = "uf"
num_points_per_task = 5000
########################
# %%
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
data_x = data_x.reshape((data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3]))
data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]

#%%
slot_fold = range(2)
shift_fold = range(1,2,1)
n_trees=[5,10,15,20,25,30,35,40,45]
iterable = product(shift_fold,slot_fold)
acc = np.zeros(len(n_trees),dtype=float)

for ii,ntree in enumerate(n_trees):
    res = Parallel(n_jobs=3,verbose=1)(
        delayed(run_parallel_exp)(
                data_x, data_y, ntree, num_points_per_task, slot=slot, shift=shift
                ) for shift,slot in iterable
            )

    acc[ii] = np.mean(res)

with open('result/cifar_acc.pickle','wb') as f:
    pickle.dump(acc,f)      

# %%
