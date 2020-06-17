#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import seaborn as sns 

import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil 

import sys
sys.path.append("../../src/")
from lifelong_dnn import LifeLongDNN
from joblib import Parallel, delayed

# %%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c

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


# %%
def experiment(n_xor, n_test, reps, n_trees, max_depth, xor=1, acorn=None):
    #print(1)
    if n_xor==0:
        raise ValueError('Wake up and provide samples to train!!!')
    
    if acorn != None:
        np.random.seed(acorn)
    
    errors = np.zeros(reps,dtype=float)
    
    for i in range(reps):
        uf = LifeLongDNN()

        if xor == 1:
            xor, label_xor = generate_gaussian_parity(n_xor,cov_scale=0.1,angle_params=0)
            test_xor, test_label_xor = generate_gaussian_parity(n_test,cov_scale=0.1,angle_params=0)
        else:
            xor, label_xor = generate_gaussian_parity(n_xor,cov_scale=0.1,angle_params=np.pi/2)
            test_xor, test_label_xor = generate_gaussian_parity(n_test,cov_scale=0.1,angle_params=np.pi/2)
    
    
        uf.new_forest(xor, label_xor, n_estimators=n_trees,max_depth=max_depth)
            
        uf_task=uf.predict(test_xor, representation=0, decider=0)
        errors[i] = 1 - np.mean(uf_task == test_label_xor)
       
    return np.mean(errors,axis=0)


# %%
mc_rep = 1000
n_test = 1000
n_trees = 10
n_xor = (100*np.arange(0.5, 20, step=0.25)).astype(int)
err = np.zeros((len(n_xor),2),dtype=float)

for i,n1 in enumerate(n_xor):
    print('starting to compute %s xor\n'%n1)
    error = np.array(
        Parallel(n_jobs=40,verbose=1)(
        delayed(experiment)(n1,n_test,1,n_trees=n_trees,max_depth=ceil(log2(750))) for _ in range(mc_rep)
    )
    )
    err[i,0] = np.mean(error, axis=0)
    err[i,1] = np.std(error, axis=0, ddof=1)

with open('result/bayes_err_xor.pickle','wb') as f:
    pickle.dump(err,f)

# %%
for i,n1 in enumerate(n_xor):
    print('starting to compute %s nxor\n'%n1)
    error = np.array(
        Parallel(n_jobs=40,verbose=1)(
        delayed(experiment)(n1,n_test,1,n_trees=n_trees,max_depth=ceil(log2(750)),xor=0) for _ in range(mc_rep)
    )
    )
    err[i,0] = np.mean(error, axis=0)
    err[i,1] = np.std(error, axis=0, ddof=1)

with open('result/bayes_err_nxor.pickle','wb') as f:
    pickle.dump(err,f)

# %%
fontsize = 24
ticksize = 20
fig, ax = plt.subplots(1,1, figsize=(8,8))
colors = sns.color_palette('Set1', n_colors=2)

mean_err = unpickle('result/bayes_err_xor.pickle')
ax.plot(n_xor, mean_err[:,0], c=colors[0], label='xor')

mean_err = unpickle('result/bayes_err_nxor.pickle')
ax.plot(n_xor, mean_err[:,0], c=colors[1], label='nxor')

ax.set_ylabel('Error', fontsize=fontsize)
ax.set_xlabel('Sample Size', fontsize=fontsize)
ax.tick_params(labelsize=ticksize)
#ax.set_ylim(0.325, 0.575)
#ax.set_title("CIFAR Recruitment",fontsize=titlesize)
ax.set_xticks([500,1000,1500,2000])
ax.set_yticks([0.1,0.125,0.15,0.175,0.2])

ax.legend(loc='upper left',fontsize=16,frameon=False)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig('result/bayes_error.pdf')

# %%
