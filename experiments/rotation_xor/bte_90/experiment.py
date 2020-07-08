import matplotlib.pyplot as plt
import random
import pickle
from skimage.transform import rotate
from scipy import ndimage
from skimage.util import img_as_ubyte
from joblib import Parallel, delayed
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.ensemble.forest import _generate_sample_indices
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from itertools import product

from joblib import Parallel, delayed
from multiprocessing import Pool

import sys
sys.path.append("../../../src")

from lifelong_dnn import LifeLongDNN

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


def LF_experiment(angle, reps=1, ntrees=10, acorn=None):
    
    errors = np.zeros(2)
    
    for rep in range(reps):
        print("Starting Rep {} of Angle {}".format(rep, angle))
        X_base_train, y_base_train = generate_gaussian_parity(n = 100, angle_params = 0, acorn=rep)
        X_base_test, y_base_test = generate_gaussian_parity(n = 10000, angle_params = 0, acorn=rep)
        X_rotated_train, y_rotated_train = generate_gaussian_parity(n = 100, angle_params = angle, acorn=rep)
        
    
        lifelong_forest = LifeLongDNN(model = "uf", parallel = True)
        lifelong_forest.new_forest(X_base_train, y_base_train, n_estimators=ntrees)
        lifelong_forest.new_forest(X_rotated_train, y_rotated_train, n_estimators=ntrees)

        all_predictions_test=lifelong_forest.predict(X_base_test, representation='all', decider=0)
        base_predictions_test=lifelong_forest.predict(X_base_test, representation=0, decider=0)

        errors[1] = errors[1]+(1 - np.mean(all_predictions_test == y_base_test))
        errors[0] = errors[0]+(1 - np.mean(base_predictions_test == y_base_test))

    
    errors = errors/reps
    print("Errors For Angle {}: {}".format(angle, errors))
    with open('results/angle_'+str(angle)+'.pickle', 'wb') as f:
        pickle.dump(errors, f, protocol = 2)
        

### MAIN HYPERPARAMS ###
granularity = 1
reps = 100
########################


def perform_angle(angle):
    LF_experiment(angle, reps=reps, ntrees=10)

angles = np.arange(0,90 + granularity,granularity)
Parallel(n_jobs=-1)(delayed(LF_experiment)(angle, reps=reps, ntrees=10) for angle in angles)