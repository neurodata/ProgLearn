#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from itertools import product
import pandas as pd
import seaborn as sns
import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# %%
file_to_load = 'result/res.pickle'
res = unpickle(file_to_load)
reps = len(res)

task_acc_single = np.zeros((10,reps),dtype=float)
task_acc_uf = np.zeros((10,reps),dtype=float)
task_acc_l2f= np.zeros((10,reps),dtype=float)

for i in range(reps):
    task_acc_single[:,i], task_acc_uf[:,i], task_acc_l2f[:,i] = res[i]

#%%
fontsize = 24
ticksize = 20
fig, ax = plt.subplots(1,1, figsize=(8,8))
colors = sns.color_palette('Set1', n_colors=3)

mean_acc = np.mean(task_acc_l2f,axis=1)
std_acc = np.std(task_acc_l2f,axis=1,ddof=1)
    
ax.plot(np.arange(1, 11), mean_acc, c=colors[0], label='L2F')
ax.fill_between(np.arange(1, 11), 
        mean_acc+ 1.96*np.array(std_acc), 
        mean_acc - 1.96*np.array(std_acc), 
        where=mean_acc + 1.96*np.array(std_acc) >= mean_acc - 1.96*np.array(std_acc), 
        facecolor=colors[0], 
        alpha=0.15,
        interpolate=False)

mean_acc = np.mean(task_acc_single,axis=1)
std_acc = np.std(task_acc_single,axis=1,ddof=1)

ax.plot(np.arange(1, 11), mean_acc, c=colors[1], label='UF fixed tree')
ax.fill_between(np.arange(1, 11), 
        mean_acc+ 1.96*np.array(std_acc), 
        mean_acc - 1.96*np.array(std_acc), 
        where=mean_acc + 1.96*np.array(std_acc) >= mean_acc - 1.96*np.array(std_acc), 
        facecolor=colors[1], 
        alpha=0.15,
        interpolate=False)


mean_acc = np.mean(task_acc_uf,axis=1)
std_acc = np.std(task_acc_uf,axis=1,ddof=1)

colors = sns.color_palette('Set1', n_colors=3)
    
ax.plot(np.arange(1, 11), mean_acc, c=colors[2], label='UF increasing tree')
ax.fill_between(np.arange(1, 11), 
        mean_acc+ 1.96*np.array(std_acc), 
        mean_acc - 1.96*np.array(std_acc), 
        where=mean_acc + 1.96*np.array(std_acc) >= mean_acc - 1.96*np.array(std_acc), 
        facecolor=colors[2], 
        alpha=0.15,
        interpolate=False)


#ax.set_title('CIFAR Recruitment Experiment', fontsize=30)
ax.set_ylabel('Accuracy', fontsize=fontsize)
ax.set_xlabel('Task Number', fontsize=fontsize)
ax.tick_params(labelsize=ticksize)
#ax.set_ylim(0.325, 0.575)
#ax.set_title("CIFAR Recruitment",fontsize=titlesize)
ax.set_xticks([1,2,3,4,5,6,7,8,9,10])
ax.set_yticks([0.1,0.2, 0.3, 0.4,0.5])

ax.legend(loc='upper left',fontsize=12,frameon=False)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig('result/batch_exp.pdf')
# %%
