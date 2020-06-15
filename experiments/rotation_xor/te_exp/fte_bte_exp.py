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
    
    l2f = LifeLongDNN(model = "uf", parallel = True)
    
    X_train_task0, y_train_task0 = generate_gaussian_parity(n = num_task_1_data, angle_params = 0, acorn = 1)
    X_train_task1, y_train_task1 = generate_gaussian_parity(n = 100, angle_params = 10, acorn = 1)
    X_test_task0, y_test_task0 = generate_gaussian_parity(n = 10000, angle_params = 0, acorn = 2)
    

    l2f.new_forest(
        X_train_task0, 
        y_train_task0, 
        n_estimators=10)
    llf_task=l2f.predict(X_test_task0, representation=0, decider=0)
    single_task_accuracy = np.nanmean(llf_task == y_test_task0)
    single_task_error = 1 - single_task_accuracy


    l2f.new_forest(
        X_train_task1, 
        y_train_task1, 
        n_estimators=10)
    
    llf_task=l2f.predict(X_test_task0, representation="all", decider=0)
    double_task_accuracy = np.nanmean(llf_task == y_test_task0)
    double_task_error = 1 - double_task_accuracy
    
    if double_task_error == 0 or single_task_error == 0:
        te = 1
    else:
        te = (single_task_error + 1e-6) / (double_task_error + 1e-6)

    df = pd.DataFrame()
    df['te'] = [te]

    file_to_save = 'result/'+str(num_task_1_data)+'_'+str(rep)+'.pickle'
    with open(file_to_save, 'wb') as f:
        pickle.dump(df, f)

#%%
def generate_gaussian_parity(n, mean=np.array([0, 0]), cov_scale=1, angle_params=None, k=1, acorn=None):
        
    blob = np.random.multivariate_normal(mean, cov_scale * np.eye(len(mean)), 
                                                 size=n)
    
    Y = np.logical_xor(blob[:, 0] > 0, blob[:, 1] > 0)
                
    if len(mean) == 2:
        X = np.zeros_like(blob)
        X[:, 0] = blob[:, 0] * np.cos(angle_params * np.pi / 180) + blob[:, 1] * np.sin(angle_params * np.pi / 180)
        X[:, 1] = -blob[:, 0] * np.sin(angle_params * np.pi / 180) + blob[:, 1] * np.cos(angle_params * np.pi / 180)
    else:
        raise ValueError('d=%i not implemented!'%(d))
       
    return X, Y.astype(int)


#%%
num_task_1_data_ra=range(10, 5000, 50)
reps = range(500)
iterable = product(num_task_1_data_ra, reps)
Parallel(n_jobs=-1,verbose=1)(delayed(LF_experiment)(num_task_1_data, rep) for num_task_1_data, rep in iterable)

    
        

# %%
