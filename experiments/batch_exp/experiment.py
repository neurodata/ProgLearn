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

#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
#%%
def LF_experiment(train_x, train_y, test_x, test_y, ntrees, shift, slot, num_points_per_task, acorn=None):
       
    uf_accuracies = np.zeros(10,dtype=float)
    rf_accuracies = np.zeros(10,dtype=float)
    single_task_accuracies = np.zeros(10,dtype=float)
    l2f_accuracies = np.zeros(10,dtype=float)

    for task_ii in range(10):
        single_task_learner = LifeLongDNN(model = "uf", parallel = True)

        if acorn is not None:
            np.random.seed(acorn)

        single_task_learner.new_forest(
            train_x[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task,:],
            train_y[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task], 
            max_depth=ceil(log2(num_points_per_task)), n_estimators=ntrees
            )
        llf_task=single_task_learner.predict(
            test_x[task_ii*1000:(task_ii+1)*1000,:], representation=0, decider=0
            )
        single_task_accuracies[task_ii] = np.mean(
                llf_task == test_y[task_ii*1000:(task_ii+1)*1000]
                )

    lifelong_forest = LifeLongDNN(model = "uf", parallel = True)
    for task_ii in range(10):
        print("Starting L2F Task {} For Fold {}".format(task_ii, shift))
        if acorn is not None:
            np.random.seed(acorn)

        lifelong_forest.new_forest(
            train_x[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task,:], 
            train_y[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task], 
            max_depth=ceil(log2(num_points_per_task)), n_estimators=ntrees
            )
        
        llf_task=lifelong_forest.predict(
                test_x[task_ii*1000:(task_ii+1)*1000,:], representation='all', decider=task_ii
                )
        l2f_accuracies[task_ii] = np.mean(
                llf_task == test_y[task_ii*1000:(task_ii+1)*1000]
                )

        
    for task_ii in range(10):
        print("Starting UF Task {} For Fold {}".format(task_ii, shift))
        lifelong_forest = LifeLongDNN(model = "uf", parallel = True)
        if acorn is not None:
            np.random.seed(acorn)

        if task_ii== 0:
            train_data_x = train_x[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task,:]
            train_data_y = train_y[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task]
        else:
            train_data_x = np.concatenate(
                (
                    train_data_x,
                    train_x[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task,:]
                ),
                axis = 0
            )
            train_data_y = np.concatenate(
                (
                    train_data_y,
                    train_y[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task]
                ),
                axis = 0
            )
        lifelong_forest.new_forest(
            train_data_x, 
            train_data_y, 
            max_depth=ceil(log2(num_points_per_task*(task_ii+1))), n_estimators=(task_ii+1)*ntrees
            )
        
        llf_task=lifelong_forest.predict(
                test_x[task_ii*1000:(task_ii+1)*1000,:], representation=0, decider=0
                )
        uf_accuracies[task_ii] = np.mean(
                llf_task == test_y[task_ii*1000:(task_ii+1)*1000]
                )





    for task_ii in range(10):
        print("Starting RF Task {} For Fold {}".format(task_ii, shift))
        RF = BaggingClassifier(
                DecisionTreeClassifier(
                max_depth=ceil(log2(num_points_per_task*(task_ii+1))),
                min_samples_leaf=1,
                max_features="auto"
            ),
            n_estimators=(task_ii+1)*ntrees,
            max_samples=0.63,
            n_jobs = -1
        )

        if acorn is not None:
            np.random.seed(acorn)

        if task_ii== 0:
            train_data_x = train_x[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task,:]
            train_data_y = train_y[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task]
        else:
            train_data_x = np.concatenate(
                (
                    train_data_x,
                    train_x[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task,:]
                ),
                axis = 0
            )
            train_data_y = np.concatenate(
                (
                    train_data_y,
                    train_y[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task]
                ),
                axis = 0
            )
        RF.fit(
            train_data_x, 
            train_data_y
            )
        
        llf_task=RF.predict(
            test_x[task_ii*1000:(task_ii+1)*1000,:]
            )
        rf_accuracies[task_ii] = np.mean(
            llf_task == test_y[task_ii*1000:(task_ii+1)*1000]
            )
        print(rf_accuracies[task_ii])
        
    return single_task_accuracies,uf_accuracies,rf_accuracies,l2f_accuracies

#%%
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

#%%
def run_parallel_exp(data_x, data_y, n_trees, num_points_per_task, slot=0, shift=1):
    train_x, train_y, test_x, test_y = cross_val_data(data_x, data_y, num_points_per_task, shift=shift)
    
    single_task_accuracies,uf_accuracies,rf_accuracies,l2f_accuracies = LF_experiment(train_x, train_y, test_x, test_y, n_trees, shift, slot, num_points_per_task, acorn=12345)

    return single_task_accuracies,uf_accuracies,rf_accuracies,l2f_accuracies
#%%
### MAIN HYPERPARAMS ###
model = "uf"
num_points_per_task = 5000
########################

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
data_x = data_x.reshape((data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3]))
data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]


#%%
slot_fold = range(5000//num_points_per_task)
shift_fold = range(1,7,1)
n_trees=[10]
iterable = product(n_trees,shift_fold,slot_fold)

res = Parallel(n_jobs=1,verbose=1)(
    delayed(run_parallel_exp)(
            data_x, data_y, ntree, num_points_per_task, slot=slot, shift=shift
            ) for ntree,shift,slot in iterable
        )

with open('result/res_'+str(num_points_per_task)+'.pickle','wb') as f:
    pickle.dump(res,f)      
# %%
