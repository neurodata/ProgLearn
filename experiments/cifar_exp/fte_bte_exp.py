#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from itertools import product

import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil 

import sys
sys.path.append("../../src/")
from lifelong_dnn import LifeLongDNN
from joblib import Parallel, delayed

#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
#%%
def LF_experiment(train_x, train_y, test_x, test_y, ntrees, cv, acorn=None):
       
    m = 1000
    errors = [[] for i in range(10)]
    errors_1 = np.zeros(10,dtype=float)
    
    lifelong_forest = LifeLongDNN()
    
    for ii in range(10):
        if acorn is not None:
            np.random.seed(acorn)

        lifelong_forest.new_forest(
            train_x[ii*5000:(ii+1)*5000,:], train_y[ii*5000:(ii+1)*5000], 
            max_depth=ceil(log2(5000)), n_estimators=ntrees
            )
        #lifelong_forest.new_forest(train_x[(ii-1)*5000:(ii)*5000,:], homogenize_labels(train_y[(ii-1)*5000:(ii)*5000]), n_estimators=ntrees)
        
        llf_task=lifelong_forest.predict(
            test_x[ii*1000:(ii+1)*1000,:], representation=ii, decider=ii
            )
        errors_1[ii] = 1 - np.sum(
                llf_task == test_y[ii*1000:(ii+1)*1000]
                )/m
        
        for jj in range(ii+1):
            llf_task=lifelong_forest.predict(
                test_x[jj*1000:(jj+1)*1000,:], representation='all', decider=jj
                )
            errors[ii].append(1 - np.sum(
                llf_task == test_y[jj*1000:(jj+1)*1000]
                )/m)
            #print(np.sum(llf_task == homogenize_labels(test_y[jj*1000:(jj+1)*1000]))/m)
    
    
    with open('./result/'+'LF_'+str(ntrees)+'__'+str(cv)+'.pickle', 'wb') as f:
        pickle.dump(errors, f)
        
    with open('./result/'+'LF_single_task_'+str(ntrees)+'__'+str(cv)+'.pickle', 'wb') as f:
        pickle.dump(errors_1, f)

#%%
def cross_val_data(data_x, data_y, class_idx, total_cls=100, cv=1):
    x = data_x.copy()
    y = data_y.copy()
    idx = class_idx.copy()
    
    
    for i in range(total_cls):
        indx = np.roll(idx[i],(cv-1)*100)
        
        if i==0:
            #train_x = x[indx[0:500],:]
            train_x = x[indx[0:500],:]
            test_x = x[indx[500:600],:]
            #train_y = y[indx[0:500]]
            train_y = y[indx[0:500]]
            test_y = y[indx[500:600]]
        else:
            #train_x = np.concatenate((train_x, x[indx[0:500],:]), axis=0)
            train_x = np.concatenate((train_x, x[indx[0:500],:]), axis=0)
            test_x = np.concatenate((test_x, x[indx[500:600],:]), axis=0)
            #train_y = np.concatenate((train_y, y[indx[0:500]]), axis=0)
            train_y = np.concatenate((train_y, y[indx[0:500]]), axis=0)
            test_y = np.concatenate((test_y, y[indx[500:600]]), axis=0)
        
        
    return train_x, train_y, test_x, test_y

#%%
def run_parallel_exp(data_x, data_y, class_idx, n_trees, total_cls=100, cv=1):
    train_x, train_y, test_x, test_y = cross_val_data(data_x, data_y, class_idx, cv=cv)
    LF_experiment(train_x, train_y, test_x, test_y, n_trees, cv, acorn=12345)

#%%
n_tasks = 10
train_file = '/data/Jayanta/continual-learning/train'
unpickled_train = unpickle(train_file)
train_keys = list(unpickled_train.keys())
fine_labels = np.array(unpickled_train[train_keys[2]])
labels = fine_labels

test_file = '/data/Jayanta/continual-learning/test'
unpickled_test = unpickle(test_file)
test_keys = list(unpickled_test.keys())
fine_labels = np.array(unpickled_test[test_keys[2]])
labels_ = fine_labels

data_x = np.concatenate((unpickled_train[b'data'], unpickled_test[b'data']), axis=0)
data_y = np.concatenate((labels, labels_), axis=0)

class_idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]

#%%
cv_fold = range(1,7,1)
#n_trees = np.asarray([41,82,164,328,656,1312,2624])
n_trees=[41]
iterable = product(n_trees,cv_fold)

Parallel(n_jobs=-2,verbose=1)(delayed(run_parallel_exp)(data_x, data_y, class_idx, ntree, total_cls=100, cv=i) for ntree,i in iterable)

#for i in range(1,7):
#    print('doing %d fold'%i)
#    run_parallel_exp(data_x, data_y, class_idx, n_trees, total_cls=100, cv=i)

# %%
