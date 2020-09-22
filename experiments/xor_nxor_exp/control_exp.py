#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

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
def exp(n_sample, n_test, angle_params, n_trees, reps, acorn=None):
    
    if acorn != None:
        np.random.seed(acorn)
        
    error = np.zeros(reps,dtype=float)
    
    for i in range(reps):
        train, label = generate_gaussian_parity(n_sample,cov_scale=0.1,angle_params=angle_params)
        test, test_label = generate_gaussian_parity(n_test,cov_scale=0.1,angle_params=angle_params)
    
        l2f = LifeLongDNN()
        l2f.new_forest(train, label, n_estimators=n_trees, max_samples=ceil(log2(n_sample)))
    
        uf_task = l2f.predict(test, representation=0, decider=0)
        error[i] = 1 - np.sum(uf_task == test_label)/n_test
    
    return np.mean(error,axis=0), np.std(error,ddof=1,axis=0)

#%%
n_trees = range(1,50,1)
n_test = 1000
n_sample = 1500
reps = 20
error1 = np.zeros(len(n_trees),dtype=float)
error2 = np.zeros(len(n_trees),dtype=float)

'''for count,n_tree in enumerate(n_trees):
    print(count)
    error1[count],_ = exp(n_sample,n_test,angle_params=0,n_trees=n_tree,reps=reps)

for count,n_tree in enumerate(n_trees):
    error2[count],_ = exp(n_sample,n_test,angle_params=np.pi/4,n_trees=n_tree,reps=reps)'''

error1 = np.array(
                Parallel(n_jobs=-2,verbose=1)(
                delayed(exp)(n_sample,n_test,angle_params=0,n_trees=n_tree,reps=reps) for n_tree in n_trees
                )
            )

error2 = np.array(
                Parallel(n_jobs=-2,verbose=1)(
                delayed(exp)(n_sample,n_test,angle_params=np.pi/4,n_trees=n_tree,reps=reps) for n_tree in n_trees
                )
            )
    
with open('./result/control_xor.pickle','wb') as f:
    pickle.dump(error1,f)
    
with open('./result/control_nxor.pickle','wb') as f:
    pickle.dump(error2,f)
    
#%% plotting the results
n_trees = np.arange(1,50,1)

tmp1 = unpickle('./result/control_xor.pickle')
tmp2 = unpickle('./result/control_nxor.pickle')

err1 = np.zeros(len(n_trees),dtype=float)
err2 = np.zeros(len(n_trees),dtype=float)

for i in range(len(n_trees)):
    err1[i] = 1-tmp1[i][0]
    err2[i] = 1-tmp2[i][0]


fig, ax = plt.subplots(1,2, figsize=(26,8))
ax[0].plot(n_trees, err1, marker='.', markersize=14, linewidth=3)
ax[1].plot(n_trees, err2, marker='.', markersize=14, linewidth=3)

ax[0].set_title('XOR',fontsize=30)
ax[0].set_xlabel('Accuracy')
ax[0].set_ylabel('Accuracy', fontsize=30)
ax[0].set_xlabel('Number of trees', fontsize=30)
ax[0].tick_params(labelsize=27.5)
#ax.set_xticks(rotation=90)

ax[1].set_title('RXOR',fontsize=30)
ax[1].set_xlabel('Accuracy')
ax[1].set_ylabel('Accuracy', fontsize=30)
ax[1].set_xlabel('Number of trees', fontsize=30)
ax[1].tick_params(labelsize=27.5)

for i in range(1,50,10):
    ax[0].axvline(x = i, linewidth=1.5,alpha=0.5, color='k')
    ax[1].axvline(x = i, linewidth=1.5,alpha=0.5, color='k')
    
plt.savefig('./result/control_xor_rxor.png',dpi=500)
plt.savefig('./result/control_xor_rxor.pdf',dpi=500)

# %%
