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

def generate_data(task_ii, num_task_1_data):
    if task_ii == 0:
        X_train, y_train = generate_gaussian_parity(n = num_task_1_data, angle_params = task_to_angle_map[task_ii])
    else:
        X_train, y_train = generate_gaussian_parity(n = 100, angle_params = task_to_angle_map[task_ii])

    X_test, y_test = generate_gaussian_parity(n = 10000, angle_params = task_to_angle_map[task_ii], acorn = 0)
    
    return X_train, X_test, y_train, y_test
    
#%%
def LF_experiment(num_task_1_data, rep):
       
    df = pd.DataFrame()
    single_task_accuracies = np.zeros(10,dtype=float)
    tasks = []
    base_tasks = []
    reps = []
    accuracies_across_tasks = []

    for task_ii in range(10):
        X_train, X_test, y_train, y_test = generate_data(task_ii, num_task_1_data)
           
        single_task_learner = LifeLongDNN(model = "uf", parallel = True)

        single_task_learner.new_forest(X_train, y_train, n_estimators=10)
        llf_task=single_task_learner.predict(X_test, representation=0, decider=0)
        single_task_accuracies[task_ii] = np.mean(llf_task == y_test)

    lifelong_forest = LifeLongDNN(model = model, parallel = True if model == "uf" else False)
    for task_ii in range(10):
        print("Starting Task {}".format(task_ii))
            
        X_train, _, y_train, _ = generate_data(task_ii, num_task_1_data)

        lifelong_forest.new_forest(
            X_train, 
            y_train, 
            n_estimators=10
            )
        
        for task_jj in range(task_ii+1):
            _, X_test, _, y_test = generate_data(task_jj, num_task_1_data)
            llf_task=lifelong_forest.predict(
                X_test, representation='all', decider=task_jj
                )
            
            tasks.append(task_jj+1)
            base_tasks.append(task_ii+1)
            accuracies_across_tasks.append(np.mean(
                llf_task == y_test
                ))
            reps.append(rep)
            
    df['task'] = tasks
    df['base_task'] = base_tasks
    df['accuracy'] = accuracies_across_tasks

    df_single_task = pd.DataFrame()
    df_single_task['task'] = range(1, 11)
    df_single_task['accuracy'] = single_task_accuracies

    summary = (df,df_single_task)
    file_to_save = 'result/result/'+str(num_task_1_data)+'_'+str(rep)+'.pickle'
    with open(file_to_save, 'wb') as f:
        pickle.dump(summary, f)

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
### MAIN HYPERPARAMS ###
model = "uf"
########################

task_to_angle_map = {}
task_to_angle_map[0] = 0
for task, angle in enumerate(np.arange(22.5, 45 + (45 - 22.5) / 9, (45 - 22.5) / 9)):
    task_to_angle_map[task] = angle

#%%
num_task_1_data_ra=[50, 100, 200, 500]
reps = range(10)
iterable = product(num_task_1_data_ra, reps)
Parallel(n_jobs=-2,verbose=1)(delayed(LF_experiment)(num_task_1_data, rep) for num_task_1_data, rep in iterable)

    
        

# %%
