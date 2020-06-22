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

def stratified_scatter(te_dict,axis_handle,s,color):
    algo = list(te_dict.keys())
    total_alg = len(algo)

    total_points = len(te_dict[algo[0]])

    pivot_points = np.arange(-.25, (total_alg+1)*1, step=1)
    interval = .7/(total_points-1)

    for algo_no,alg in enumerate(algo):
        for no,points in enumerate(te_dict[alg]):
            axis_handle.scatter(
                pivot_points[algo_no]+interval*no,
                te_dict[alg][no],
                s=s,
                c=color[algo_no]
                )


#%%
task_num=15
#filename = 'result/PL_llf_language.p'
#filename = 'result/satori_llf_overlap_random_batch.p'
#filename = 'result/satori_llf_nonoverlap_batch.p'
filename = 'result/PL_llf_language_random_overlap_batches.p'
res = unpickle(filename)

# %%
reps = 15
count = 0
bte_tmp = [[] for _ in range(reps)]
fte_tmp = [[] for _ in range(reps)] 
te_tmp = [[] for _ in range(reps)] 

for seed,single_task_err,multitask_err in res:
    single_err, err = get_error_matrix(single_task_err,multitask_err,task_num=task_num)
    fte, bte, te = get_fte_bte(err,single_err,task_num=task_num)

    bte_tmp[count].extend(bte)
    fte_tmp[count].extend(fte)
    te_tmp[count].extend(te)
    count+=1

btes = calc_mean_bte(bte_tmp,task_num=task_num,reps=reps)
ftes = calc_mean_fte(fte_tmp,task_num=task_num,reps=reps)
tes = calc_mean_te(te_tmp,task_num=task_num,reps=reps)


# %%
fig = plt.figure(constrained_layout=True,figsize=(16,8))
gs = fig.add_gridspec(8, 16)

fontsize=25
ticksize=22
legendsize=14

ax = fig.add_subplot(gs[:7,:7])
ax.plot(np.arange(1,task_num+1), fte, color='r', marker='.', markersize=12, linewidth=3)

ax.set_xticks(np.arange(1,11))
#ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3,1.4])
#ax.set_ylim(0.89, 1.41)
ax.tick_params(labelsize=ticksize)

ax.set_ylabel('Forward Transfer Efficiency (FTE)', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,task_num, colors='grey', linestyles='dashed',linewidth=1.5)


ax = fig.add_subplot(gs[:7,8:15])
for i in range(task_num-1):

    et = np.zeros((1,task_num-i))

    ns = np.arange(i + 1, task_num + 1)
    ax.plot(ns, btes[i], marker='.', markersize=8, color='r', linewidth = 3)
ax.plot(task_num, btes[task_num-1], marker='.', markersize=8, color='r', linewidth = 3)

ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('Backward Transfer Efficiency (BTE)', fontsize=fontsize)

#ax.set_yticks([.4,.6,.8,.9,1, 1.1,1.2])
ax.set_xticks(np.arange(1,task_num+1))
#ax.set_ylim(0.99, 1.2)
ax.tick_params(labelsize=ticksize)
#ax[0][1].grid(axis='x')

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,task_num, colors='grey', linestyles='dashed',linewidth=1.5)

plt.savefig('result/PL_llf_overlap.pdf')
#plt.savefig('result/satori_overlap.pdf')
#plt.savefig('result/satori_nonoverlap.pdf')
# %%
