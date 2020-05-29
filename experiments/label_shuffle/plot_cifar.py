#%%
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
from itertools import product
import seaborn as sns

### MAIN HYPERPARAMS ###
ntrees = 10
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
        bte.append(err[i] / err[0])
    
    return bte

def calc_mean_bte(btes,task_num=10,reps=6):
    mean_bte = []
    

    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(btes[i])
        
        tmp=tmp/reps
        mean_bte.extend(tmp)
            
    return mean_bte       

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
            err.extend(
             1 - np.array(
                 multitask_df[multitask_df['task']==ii+1]['task_1_accuracy']
             )
            )
        bte = get_bte(err)
        
        bte_tmp[count].extend(bte)
        count+=1
    
bte = np.mean(bte_tmp, axis = 0)

#%%
sns.set()

n_tasks=10
clr = ["#e41a1c", "#a65628", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#CCCC00"]
#c = sns.color_palette(clr, n_colors=len(clr))

fontsize=22
ticksize=20

plt.plot(range(1, n_tasks + 1), bte)
    
plt.xlabel('Number of tasks seen', fontsize=fontsize)
plt.ylabel('BTE', fontsize=fontsize)
plt.tick_params(labelsize=ticksize)
plt.hlines(1, 1,n_tasks, colors='grey', linestyles='dashed',linewidth=1.5)

plt.savefig('./result/figs/fig_trees'+str(ntrees)+"__"+model+'.pdf',dpi=300)

# %%
