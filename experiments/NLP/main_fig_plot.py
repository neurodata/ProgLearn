#%%
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib

#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_error_matrix(single_task_err,multitask_err,task_num):
    err = [[] for _ in range(task_num)]
    single_err = np.zeros(task_num,dtype=float)

    for ii in range(task_num):
        single_err[ii] = 1-single_task_err[ii]
        for jj in range(ii+1):
            #print(multitask_err[jj])
            tmp =  1 - multitask_err[jj][ii-jj]
            if tmp==0:
                tmp = 1e-6

            err[ii].append(tmp)

    return single_err, err

def get_fte_bte(err, single_err, task_num):
    bte = [[] for i in range(task_num)]
    te = [[] for i in range(task_num)]
    fte = []
    
    for i in range(task_num):
        for j in range(i,task_num):
            #print(err[j][i],j,i)
            bte[i].append(err[i][i]/err[j][i])
            te[i].append(single_err[i]/err[j][i])
                
    for i in range(task_num):
        fte.append(single_err[i]/err[i][i])
            
            
    return fte,bte,te

def calc_mean_bte(btes,task_num=10,reps=6):
    mean_bte = [[] for i in range(task_num)]


    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(btes[i][j])
        
        tmp=tmp/reps
        mean_bte[j].extend(tmp)
            
    return mean_bte     

def calc_mean_te(tes,task_num=10,reps=6):
    mean_te = [[] for i in range(task_num)]

    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(tes[i][j])
        
        tmp=tmp/reps
        mean_te[j].extend(tmp)
            
    return mean_te 

def calc_mean_fte(ftes,task_num=10,reps=6):
    fte = np.asarray(ftes)
    
    return list(np.mean(np.asarray(fte),axis=0))


#%%
task_num1=15
task_num2=10

filename1 = 'result/PL_llf_language_random_overlap_batches.p'
filename2 = 'result/satori_llf_overlap_random_batch.p'
res1 = unpickle(filename1)
res2 = unpickle(filename2)
# %%
reps = 15
count = 0
bte_tmp = [[] for _ in range(reps)]
fte_tmp = [[] for _ in range(reps)] 
te_tmp = [[] for _ in range(reps)] 

for seed,single_task_err,multitask_err in res1:
    single_err, err = get_error_matrix(single_task_err,multitask_err,task_num=task_num1)
    fte, bte, te = get_fte_bte(err,single_err,task_num=task_num1)

    bte_tmp[count].extend(bte)
    fte_tmp[count].extend(fte)
    te_tmp[count].extend(te)
    count+=1

btes1 = calc_mean_bte(bte_tmp,task_num=task_num1,reps=reps)
ftes1 = calc_mean_fte(fte_tmp,task_num=task_num1,reps=reps)
tes1 = calc_mean_te(te_tmp,task_num=task_num1,reps=reps)

# %%
reps = 10
count = 0
bte_tmp = [[] for _ in range(reps)]
fte_tmp = [[] for _ in range(reps)] 
te_tmp = [[] for _ in range(reps)] 

for seed,single_task_err,multitask_err in res2:
    single_err, err = get_error_matrix(single_task_err,multitask_err,task_num=task_num2)
    fte, bte, te = get_fte_bte(err,single_err,task_num=task_num2)

    bte_tmp[count].extend(bte)
    fte_tmp[count].extend(fte)
    te_tmp[count].extend(te)
    count+=1

btes2 = calc_mean_bte(bte_tmp,task_num=task_num2,reps=reps)
ftes2 = calc_mean_fte(fte_tmp,task_num=task_num2,reps=reps)
tes2 = calc_mean_te(te_tmp,task_num=task_num2,reps=reps)


# %%
fig = plt.figure(constrained_layout=True,figsize=(16,8))
gs = fig.add_gridspec(8, 16)

fontsize=25
ticksize=22
legendsize=14

ax = fig.add_subplot(gs[:7,:7])
for i in range(task_num1-1):

    et = np.zeros((1,task_num1-i))

    ns = np.arange(i + 1, task_num1 + 1)
    ax.plot(ns, btes1[i], marker='.', markersize=8, color='r', linewidth = 3)
ax.plot(task_num1, btes1[task_num1-1], marker='.', markersize=8, color='r', linewidth = 3)

ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('Backward Transfer Efficiency', fontsize=fontsize)

#ax.set_yticks([.4,.6,.8,.9,1, 1.1,1.2])
ax.set_xticks(np.arange(1,task_num1+1,2))
#ax.set_ylim(0.99, 1.2)
ax.tick_params(labelsize=ticksize)
#ax[0][1].grid(axis='x')
ax.set_ylim([.5,1.5])
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,task_num1, colors='grey', linestyles='dashed',linewidth=1.5)

#######################################
ax = fig.add_subplot(gs[:7,8:15])
for i in range(task_num2-1):

    et = np.zeros((1,task_num2-i))

    ns = np.arange(i + 1, task_num2 + 1)
    ax.plot(ns, btes2[i], marker='.', markersize=8, color='r', linewidth = 3)
ax.plot(task_num2, btes2[task_num2-1], marker='.', markersize=8, color='r', linewidth = 3)

ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('Backward Transfer Efficiency', fontsize=fontsize)

ax.set_yticks([0.98,1,1.02,1.04,1.06])
ax.set_xticks(np.arange(1,task_num2+1,2))
#ax.set_ylim(0.99, 1.2)
ax.tick_params(labelsize=ticksize)
#ax[0][1].grid(axis='x')
ax.set_ylim([.98,1.06])
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,task_num2, colors='grey', linestyles='dashed',linewidth=1.5)

plt.savefig('result/nlp_exp.pdf')
# %%
