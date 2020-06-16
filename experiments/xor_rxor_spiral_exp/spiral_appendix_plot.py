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

def generate_spirals(N, D=2, K=5, noise = 0.5, acorn = None, density=0.3):

    #N number of poinst per class
    #D number of features, 
    #K number of classes
    X = []
    Y = []
    if acorn is not None:
        np.random.seed(acorn)
    
    if K == 2:
        turns = 2
    elif K==3:
        turns = 2.5
    elif K==5:
        turns = 3.5
    elif K==7:
        turns = 4.5
    else:
        print ("sorry, can't currently surpport %s classes " %K)
        return
    
    mvt = np.random.multinomial(N, 1/K * np.ones(K))
    
    if K == 2:
#         r = np.linspace(0.01, 1, N)
        r = np.random.uniform(0,1,size=int(N/K))
        r = np.sort(r)
        t = np.linspace(0,  np.pi* 4 * turns/K, int(N/K)) + noise * np.random.normal(0, density, int(N/K))
        dx = r * np.cos(t)
        dy = r* np.sin(t)

        X.append(np.vstack([dx, dy]).T )
        X.append(np.vstack([-dx, -dy]).T)
        Y += [0] * int(N/K) 
        Y += [1] * int(N/K)
    else:    
        for j in range(1, K+1):
            r = np.linspace(0.01, 1, int(mvt[j-1]))
            t = np.linspace((j-1) * np.pi *4 *turns/K,  j* np.pi * 4* turns/K, int(mvt[j-1])) + noise * np.random.normal(0, density, int(mvt[j-1]))
            dx = r * np.cos(t)
            dy = r* np.sin(t)

            dd = np.vstack([dx, dy]).T        
            X.append(dd)
            #label
            Y += [j-1] * int(mvt[j-1])
    return np.vstack(X), np.array(Y).astype(int)


def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c

#%% Plotting the result
#mc_rep = 50
mean_error = unpickle('result/mean_spiral.pickle')
std_error = unpickle('result/std_spiral.pickle')

spiral3 = (100*np.arange(0.5, 7.25, step=0.25)).astype(int)
spiral5 = (100*np.arange(0.5, 7.50, step=0.25)).astype(int)

n1s = spiral3
n2s = spiral5

ns = np.concatenate((n1s, n2s + n1s[-1]))
ls=['-', '--']
algorithms = ['Uncertainty Forest', 'Lifelong Forest']


TASK1='3 spirals'
TASK2='5 spirals'

fontsize=30
labelsize=28

colors = sns.color_palette("Set1", n_colors = 2)

fig = plt.figure(constrained_layout=True,figsize=(21,14))
gs = fig.add_gridspec(14, 21)
ax1 = fig.add_subplot(gs[7:,:6])
# for i, algo in enumerate(algorithms):
ax1.plot(ns, mean_error[0], label=algorithms[0], c=colors[1], ls=ls[np.sum(0 > 1).astype(int)], lw=3)
#ax1.fill_between(ns, 
#        mean_error[0] + 1.96*std_error[0], 
#        mean_error[0] - 1.96*std_error[0], 
#        where=mean_error[0] + 1.96*std_error[0] >= mean_error[0] - 1.96*std_error[0], 
#        facecolor=colors[1], 
#        alpha=0.15,
#        interpolate=True)

ax1.plot(ns, mean_error[1], label=algorithms[1], c=colors[0], ls=ls[np.sum(1 > 1).astype(int)], lw=3)
#ax1.fill_between(ns, 
#        mean_error[1] + 1.96*std_error[1, ], 
#        mean_error[1] - 1.96*std_error[1, ], 
#        where=mean_error[1] + 1.96*std_error[1] >= mean_error[1] - 1.96*std_error[1], 
#        facecolor=colors[0], 
#        alpha=0.15,
#        interpolate=True)

ax1.set_ylabel('Generalization Error (%s)'%(TASK1), fontsize=fontsize)
ax1.legend(loc='upper right', fontsize=20, frameon=False)
ax1.set_ylim(0.2, 0.61)
ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
ax1.set_yticks([0.2, 0.4,.6])
ax1.set_xticks([250,750,1500])
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
ax1.set_title('3 spirals', fontsize=30)

right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)

ax1.text(150, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)

#plt.tight_layout()

#plt.savefig('./result/figs/generalization_error_xor.pdf',dpi=500)

#####
mean_error = unpickle('result/mean_spiral.pickle')
std_error = unpickle('result/std_spiral.pickle')

algorithms = ['Uncertainty Forest', 'Lifelong Forest']

TASK1='3 spirals'
TASK2='5 spirals'

ax1 = fig.add_subplot(gs[7:,7:13])
# for i, algo in enumerate(algorithms):
ax1.plot(ns[len(n1s):], mean_error[2, len(n1s):], label=algorithms[0], c=colors[1], ls=ls[1], lw=3)
#ax1.fill_between(ns[len(n1s):], 
#        mean_error[2, len(n1s):] + 1.96*std_error[2, len(n1s):], 
#        mean_error[2, len(n1s):] - 1.96*std_error[2, len(n1s):], 
#        where=mean_error[2, len(n1s):] + 1.96*std_error[2, len(n1s):] >= mean_error[2, len(n1s):] - 1.96*std_error[2, len(n1s):], 
#        facecolor=colors[1], 
#        alpha=0.15,
#        interpolate=True)

ax1.plot(ns[len(n1s):], mean_error[3, len(n1s):], label=algorithms[1], c=colors[0], ls=ls[1], lw=3)
#ax1.fill_between(ns[len(n1s):], 
#        mean_error[3, len(n1s):] + 1.96*std_error[3, len(n1s):], 
#        mean_error[3, len(n1s):] - 1.96*std_error[3, len(n1s):], 
#        where=mean_error[3, len(n1s):] + 1.96*std_error[3, len(n1s):] >= mean_error[3, len(n1s):] - 1.96*std_error[3, len(n1s):], 
#        facecolor=colors[0], 
#        alpha=0.15,
#        interpolate=True)

ax1.set_ylabel('Generalization Error (%s)'%(TASK2), fontsize=fontsize)
ax1.legend(loc='upper right', fontsize=20, frameon=False)
#         ax1.set_ylim(-0.01, 0.22)
ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
ax1.set_yticks([0.5, 0.6, 0.7])
#ax1.set_yticks([0.15, 0.2])
ax1.set_xticks([250,750,1500])
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")

ax1.set_ylim(0.5, 0.71)

ax1.set_title('5 spirals', fontsize=30)
right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)

# ax1.set_ylim(0.14, 0.36)
ax1.text(150, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)

#plt.tight_layout()

#plt.savefig('./result/figs/generalization_error_nxor.pdf',dpi=500)

#####
mean_error = unpickle('result/mean_te_spiral.pickle')
std_error = unpickle('result/std_te_spiral.pickle')

algorithms = ['Backward Transfer', 'Forward Transfer']

TASK1='3 spirals'
TASK2='5 spirals'

ax1 = fig.add_subplot(gs[7:,14:])

ax1.plot(ns, mean_error[0], label=algorithms[0], c=colors[0], ls=ls[0], lw=3)
#ax1.fill_between(ns, 
#        mean_error[0] + 1.96*std_error[0], 
#        mean_error[0] - 1.96*std_error[0], 
#        where=mean_error[1] + 1.96*std_error[0] >= mean_error[0] - 1.96*std_error[0], 
#        facecolor=colors[0], 
#        alpha=0.15,
#        interpolate=True)

ax1.plot(ns[len(n1s):], mean_error[1, len(n1s):], label=algorithms[1], c=colors[0], ls=ls[1], lw=3)
#ax1.fill_between(ns[len(n1s):], 
#        mean_error[1, len(n1s):] + 1.96*std_error[1, len(n1s):], 
#        mean_error[1, len(n1s):] - 1.96*std_error[1, len(n1s):], 
#        where=mean_error[1, len(n1s):] + 1.96*std_error[1, len(n1s):] >= mean_error[1, len(n1s):] - 1.96*std_error[1, len(n1s):], 
#        facecolor=colors[0], 
#        alpha=0.15,
#        interpolate=True)

ax1.set_ylabel('Transfer Efficiency', fontsize=fontsize)
ax1.legend(loc='upper right', fontsize=20, frameon=False)
ax1.set_ylim(.92, 1.1)
ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
ax1.set_yticks([.92,1,1.08])
ax1.set_xticks([250,750,1500])
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)
ax1.hlines(1, 50,1500, colors='gray', linestyles='dashed',linewidth=1.5)

ax1.text(150, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)

#plt.tight_layout()

#plt.savefig('./result/figs/TE.pdf',dpi=500)

#####
colors = sns.color_palette('Dark2', n_colors=5)

X, Y = generate_spirals(750, 2, 3, noise = 2.5)
Z, W = generate_spirals(750, 2, 5, noise = 2.5)

ax = fig.add_subplot(gs[:6,4:10])
ax.scatter(X[:, 0], X[:, 1], c=get_colors(colors, Y), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('3 spirals', fontsize=30)

plt.tight_layout()
ax.axis('off')
#plt.savefig('./result/figs/gaussian-xor.pdf')

###
colors = sns.color_palette('Dark2', n_colors=5)

ax = fig.add_subplot(gs[:6,11:16])
ax.scatter(Z[:, 0], Z[:, 1], c=get_colors(colors, W), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('5 spirals', fontsize=30)
ax.axis('off')
#plt.tight_layout()
plt.savefig('./result/figs/spiral_exp.pdf')

# %%
