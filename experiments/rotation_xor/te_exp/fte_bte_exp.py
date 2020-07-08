#%%
import random
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd

import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil 

import sys
sys.path.append("../../../src/")
from lifelong_dnn import LifeLongDNN
from joblib import Parallel, delayed
from multiprocessing import Pool

#%%
def LF_experiment(num_task_1_data, rep):
    
    l2f = LifeLongDNN(model = "uf", parallel = False)
    
    X_train_task0, y_train_task0 = generate_gaussian_parity(n = num_task_1_data, angle_params = 0, acorn=rep)
    X_train_task1, y_train_task1 = generate_gaussian_parity(n = 100, angle_params = 10, acorn=rep)
    X_test_task0, y_test_task0 = generate_gaussian_parity(n = 10000, angle_params = 0, acorn=rep)
    

    l2f.new_forest(
        X_train_task0, 
        y_train_task0, 
        n_estimators=10,
        parallel=False
        )
    llf_task=l2f.predict(X_test_task0, representation=0, decider=0)
    single_task_accuracy = np.nanmean(llf_task == y_test_task0)
    single_task_error = 1 - single_task_accuracy


    l2f.new_forest(
        X_train_task1, 
        y_train_task1, 
        n_estimators=10,
        parallel=False)
    
    llf_task=l2f.predict(X_test_task0, representation="all", decider=0)
    double_task_accuracy = np.nanmean(llf_task == y_test_task0)
    double_task_error = 1 - double_task_accuracy
    
    if double_task_error == 0 or single_task_error == 0:
        te = 1
    else:
        te = (single_task_error + 1e-6) / (double_task_error + 1e-6)

    df = pd.DataFrame()
    df['te'] = [te]

    print('n = {}, te = {}'.format(num_task_1_data, te))
    file_to_save = 'result/'+str(num_task_1_data)+'_'+str(rep)+'.pickle'
    with open(file_to_save, 'wb') as f:
        pickle.dump(df, f)

#%%
def generate_2d_rotation(theta=0, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
    
    R = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    
    return R

def generate_gaussian_parity(n, mean=np.array([-1, -1]), cov_scale=1, angle_params=None, k=1, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
        
    d = len(mean)
    
    if mean[0] == -1 and mean[1] == -1:
        mean = mean + 1 / 2**k
    
    mnt = np.random.multinomial(n, 1/(4**k) * np.ones(4**k))
    cumsum = np.cumsum(mnt)
    cumsum = np.concatenate(([0], cumsum))
    
    Y = np.zeros(n)
    X = np.zeros((n, d))
    
    for i in range(2**k):
        for j in range(2**k):
            temp = np.random.multivariate_normal(mean, cov_scale * np.eye(d), 
                                                 size=mnt[i*(2**k) + j])
            temp[:, 0] += i*(1/2**(k-1))
            temp[:, 1] += j*(1/2**(k-1))
            
            X[cumsum[i*(2**k) + j]:cumsum[i*(2**k) + j + 1]] = temp
            
            if i % 2 == j % 2:
                Y[cumsum[i*(2**k) + j]:cumsum[i*(2**k) + j + 1]] = 0
            else:
                Y[cumsum[i*(2**k) + j]:cumsum[i*(2**k) + j + 1]] = 1
                
    if d == 2:
        if angle_params is None:
            angle_params = np.random.uniform(0, 2*np.pi)
            
        R = generate_2d_rotation(angle_params)
        X = X @ R
        
    else:
        raise ValueError('d=%i not implemented!'%(d))
       
    return X, Y.astype(int)

#%%
num_task_1_data_ra=(2**np.arange(np.log2(60), np.log2(5010)+1, 0.25)).astype('int')
reps = range(1000)
iterable = product(num_task_1_data_ra, reps)
Parallel(n_jobs=-1,verbose=1)(delayed(LF_experiment)(num_task_1_data, rep) for num_task_1_data, rep in iterable)

    
        

# %%
