#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import seaborn as sns 
import matplotlib
import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil 

import sys
sys.path.append("../../src/")
from lifelong_dnn import LifeLongDNN

#%%
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

#%%#%% Plotting the result
#mc_rep = 50
fontsize=30
labelsize=28


fig = plt.figure(constrained_layout=True,figsize=(21,30))
gs = fig.add_gridspec(30, 21)

colors = sns.color_palette('Dark2', n_colors=2)

X, Y = generate_gaussian_parity(750, cov_scale=0.1, angle_params=0)
Z, W = generate_gaussian_parity(750, cov_scale=0.1, angle_params=np.pi/2)
P, Q = generate_gaussian_parity(750, cov_scale=0.1, angle_params=np.pi/4)

ax = fig.add_subplot(gs[:6,:6])
ax.scatter(X[:, 0], X[:, 1], c=get_colors(colors, Y), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Gaussian XOR', fontsize=30)

plt.tight_layout()
ax.axis('off')
#plt.savefig('./result/figs/gaussian-xor.pdf')

#####################
ax = fig.add_subplot(gs[:6,7:13])
ax.scatter(Z[:, 0], Z[:, 1], c=get_colors(colors, W), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Gaussian N-XOR', fontsize=30)
ax.axis('off')

#####################
ax = fig.add_subplot(gs[:6,14:20])
ax.scatter(P[:, 0], P[:, 1], c=get_colors(colors, Q), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Gaussian R-XOR', fontsize=30)
ax.axis('off')

######################
mean_error = unpickle('../xor_nxor_exp/result/mean_xor_nxor.pickle')

n_xor = (100*np.arange(0.5, 7.25, step=0.25)).astype(int)
n_nxor = (100*np.arange(0.5, 7.50, step=0.25)).astype(int)

n1s = n_xor
n2s = n_nxor

ns = np.concatenate((n1s, n2s + n1s[-1]))
ls=['-', '--']
algorithms = ['Uncertainty Forest', 'Lifelong Forest']


TASK1='XOR'
TASK2='N-XOR'

colors = sns.color_palette("Set1", n_colors = 2)

ax = fig.add_subplot(gs[7:13,2:9])

ax.plot(ns, mean_error[0], label=algorithms[0], c=colors[1], ls=ls[np.sum(0 > 1).astype(int)], lw=3)
ax.plot(ns, mean_error[1], label=algorithms[1], c=colors[0], ls=ls[np.sum(1 > 1).astype(int)], lw=3)

ax.set_ylabel('Generalization Error (%s)'%(TASK1), fontsize=fontsize)
ax.legend(loc='upper right', fontsize=20, frameon=False)
ax.set_ylim(0.1, 0.2)
ax.set_xlabel('Total Sample Size', fontsize=fontsize)
ax.tick_params(labelsize=labelsize)
ax.set_yticks([.1,0.15, 0.2])
ax.set_xticks([250,750,1500])
ax.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
ax.set_title('XOR', fontsize=30)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.text(400, np.mean(ax.get_ylim()), "%s"%(TASK1), fontsize=26)
ax.text(900, np.mean(ax.get_ylim()), "%s"%(TASK2), fontsize=26)


#######################
mean_error = unpickle('../xor_nxor_exp/result/mean_xor_nxor.pickle')

algorithms = ['Uncertainty Forest', 'Lifelong Forest']

TASK1='XOR'
TASK2='N-XOR'
ax = fig.add_subplot(gs[7:13,12:19])

ax.plot(ns[len(n1s):], mean_error[2, len(n1s):], label=algorithms[0], c=colors[1], ls=ls[1], lw=3)
ax.plot(ns[len(n1s):], mean_error[3, len(n1s):], label=algorithms[1], c=colors[0], ls=ls[1], lw=3)

ax.set_ylabel('Generalization Error (%s)'%(TASK2), fontsize=fontsize)
ax.legend(loc='upper right', fontsize=20, frameon=False)
ax.set_xlabel('Total Sample Size', fontsize=fontsize)
ax.tick_params(labelsize=labelsize)
ax.set_yticks([.1,0.15, 0.2])
ax.set_xticks([250,750,1500])
ax.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")

ax.set_ylim(0.1, 0.2)

ax.set_xlim(-10)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.text(400, np.mean(ax.get_ylim()), "%s"%(TASK1), fontsize=26)
ax.text(900, np.mean(ax.get_ylim()), "%s"%(TASK2), fontsize=26)

ax.set_title('N-XOR', fontsize=30)

##################
mean_error = unpickle('../xor_nxor_exp/result/mean_te_xor_nxor.pickle')
std_error = unpickle('../xor_nxor_exp/result/std_te_xor_nxor.pickle')

algorithms = ['Backward Transfer', 'Forward Transfer']

TASK1='XOR'
TASK2='N-XOR'

ax = fig.add_subplot(gs[15:21,2:9])

ax.plot(ns, mean_error[0], label=algorithms[0], c=colors[0], ls=ls[0], lw=3)

ax.plot(ns[len(n1s):], mean_error[1, len(n1s):], label=algorithms[1], c=colors[0], ls=ls[1], lw=3)

ax.set_title('Transfer Efficiency for XOR and N-XOR',fontsize=fontsize)
ax.set_ylabel('Transfer Efficiency', fontsize=fontsize)
ax.legend(loc='upper right', fontsize=20, frameon=False)
ax.set_ylim(.99, 1.42)
ax.set_xlabel('Total Sample Size', fontsize=fontsize)
ax.tick_params(labelsize=labelsize)
ax.set_yticks([1,1.2,1.4])
ax.set_xticks([250,750,1500])
ax.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 50,1500, colors='gray', linestyles='dashed',linewidth=1.5)

ax.text(400, np.mean(ax.get_ylim()), "%s"%(TASK1), fontsize=26)
ax.text(900, np.mean(ax.get_ylim()), "%s"%(TASK2), fontsize=26)

######################
mean_error = unpickle('./result/mean_te_xor_rxor.pickle')

algorithms = ['Backward Transfer', 'Forward Transfer']

TASK1='XOR'
TASK2='R-XOR'

ax = fig.add_subplot(gs[15:21,12:19])

ax.plot(ns, mean_error[0], label=algorithms[0], c=colors[0], ls=ls[0], lw=3)

ax.plot(ns[len(n1s):], mean_error[1, len(n1s):], label=algorithms[1], c=colors[0], ls=ls[1], lw=3)

ax.set_title('Transfer Efficiency for XOR and R-XOR',fontsize=fontsize)
ax.set_ylabel('Transfer Efficiency', fontsize=fontsize)
ax.legend(loc='upper right', fontsize=20, frameon=False)
ax.set_ylim(0.96, 1.045)
ax.set_xlabel('Total Sample Size', fontsize=fontsize)
ax.tick_params(labelsize=labelsize)
ax.set_yticks([0.96,1, 1.04])
ax.set_xticks([250,750,1500])
ax.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 50,1500, colors='gray', linestyles='dashed',linewidth=1.5)

ax.text(400, np.mean(ax.get_ylim())+.005, "%s"%(TASK1), fontsize=26)
ax.text(900, np.mean(ax.get_ylim())+.005, "%s"%(TASK2), fontsize=26)

########################################################
ax = fig.add_subplot(gs[23:29,2:9])
alg_name = ['L2F']
angles = np.arange(0,91,1)
tes = [[] for _ in range(len(alg_name))]

for algo_no,alg in enumerate(alg_name):
    for angle in angles:
        orig_error, transfer_error = pickle.load(
                open("../rotation_xor/bte_90/results/angle_" + str(angle) + ".pickle", "rb")
                )
        tes[algo_no].append(orig_error / transfer_error)

clr = ["#e41a1c"]
c = sns.color_palette(clr, n_colors=len(clr))

for alg_no,alg in enumerate(alg_name):
    if alg_no<2:
        ax.plot(angles,tes[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3)
    else:
        ax.plot(angles,tes[alg_no], c=c[alg_no], label=alg_name[alg_no])


ax.set_xticks(range(0, 90 + 15, 15))
ax.set_yticks([0.6,1.0,1.8])
ax.tick_params(labelsize=labelsize)
ax.set_xlabel('Angle of Rotation (Degrees)', fontsize=fontsize)
ax.set_ylabel('Backward Transfer Efficiency (XOR)', fontsize=fontsize)
#ax.set_title("XOR vs. Rotated-XOR", fontsize = fontsize)
ax.hlines(1,0,90, colors='grey', linestyles='dashed',linewidth=1.5)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)


ax = fig.add_subplot(gs[23:29,12:19])
te_ra = []
n1_ra = (2**np.arange(np.log2(60), np.log2(5010)+1, .25)).astype('int')
#print(n1_ra)
for n1 in n1_ra:
    te_across_reps = []
    for rep in range(1000):
        filename = '../rotation_xor/te_exp/result/'+str(n1)+'_'+str(rep)+'.pickle'
        df = unpickle(filename)
        te_across_reps.append(float(df['te']))
    te_ra.append(np.mean(te_across_reps))
    
ax.plot(n1_ra, te_ra, c="#e41a1c", linewidth = 2.6)
ax.set_xscale('log')
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#ax.set_xticks([])
ax.tick_params(labelsize=labelsize)
ax.set_xticks([100, 500, 5000])
ax.set_yticks([0.7,1.0,1.2])
#ax.text(100, 0.1, "100", fontsize=labelsize)
#ax.text(500, 0.1, "500", fontsize=labelsize)
#ax.text(2500, 0.434, "2500", fontsize=labelsize)
#ax.text(5000, 0.434, "5000", fontsize=labelsize)
#ax.text(500, 0.43, 'Number of Task 1 Training Samples', fontsize=labelsize)

ax.hlines(1, min(n1_ra), max(n1_ra), colors='grey', linestyles='dashed',linewidth=1.5)
ax.set_xlabel(r'Number of $10^\circ$-RXOR Training Samples', fontsize=fontsize)
ax.set_ylabel('Backward Transfer Efficiency (XOR)', fontsize=fontsize)
#ax.set_title("Training Set Size Effect", fontsize = fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig('./result/figs/xor_nxor_rxor_exp.pdf')

# %%
