#%%
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
from itertools import product
import seaborn as sns

### MAIN HYPERPARAMS ###
ntrees = 50
slots = 10
shifts = 6
alg_num = 1
task_num = 10
model = "uf"
########################

#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_bte(err):
    bte = []
    
    for i in range(10):
        bte.append(err[i]/err[0])
    
    return bte

def calc_mean_bte(btes,task_num=10,reps=6):
    mean_bte = [[] for i in range(task_num)]


    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(btes[i][j])
        
        tmp=tmp/reps
        mean_bte[j].extend(tmp)
            
    return mean_bte     

def calc_mean_err(err,task_num=10,reps=6):
    mean_err = [[] for i in range(task_num)]


    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(err[i][j])
        
        tmp=tmp/reps
        #print(tmp)
        mean_err[j].extend([tmp])
            
    return mean_err     

#%%
reps = slots*shifts

err_ = []

bte_tmp = [[] for _ in range(reps)]

count = 0   
for slot in range(slots):
    for shift in range(shifts):
        filename = 'result/'+model+str(ntrees)+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
        multitask_df = unpickle(filename)

        err = []

        for ii in range(10):
            err[ii].extend(
             1 - np.array(
                 multitask_df[multitask_df['task']==ii+1]['task_1_accuracy']
             )
            )
        bte = get_bte(err)
        
        bte_tmp[count].extend(bte)
        count+=1
    
bte = calc_mean_bte(bte_tmp,reps=reps)

#%%
sns.set()

n_tasks=10
clr = ["#e41a1c", "#a65628", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#CCCC00"]
#c = sns.color_palette(clr, n_colors=len(clr))

fontsize=22
ticksize=20

fig, ax = plt.subplots(1,1, figsize=(16,11.5))


for i in range(n_tasks):

    et = np.asarray(bte[i])

    ns = np.arange(i + 1, n_tasks + 1)
    ax[0][1].plot(ns, et, c='red', linewidth = 2.6)
    
ax[0][0].set_xlabel('Number of tasks seen', fontsize=fontsize)
ax[0][0].set_ylabel('BTE', fontsize=fontsize)
ax[0][0].set_yticks([.4,.6,.8,.9,1, 1.1,1.2])
#ax[0][1].set_xticks(np.arange(1,10))
ax[0][0].set_ylim(0.89, 1.15)
ax[0][0].tick_params(labelsize=ticksize)
ax[0][0].hlines(1, 1,n_tasks, colors='grey', linestyles='dashed',linewidth=1.5)

plt.savefig('./result/figs/fig_trees'+str(ntrees)+"__"+model+'.pdf',dpi=300)

# %%
