#%%
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
from itertools import product
import seaborn as sns

### MAIN HYPERPARAMS ###
ntrees = 0
slots = 10
shifts = 6
alg_num = 1
task_num = 10
model = "dnn"
########################

#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


#%%
reps = slots // 2 *shifts

btes = [[] for i in range(task_num)]

bte_tmp = [[] for _ in range(reps)]

count = 0   
for slot in range(0, slots, 2):
    for shift in range(shifts):
        filename = 'result/'+model+str(ntrees)+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
        df = unpickle(filename)

        err = 1 - df['accuracy']
        bte = err[0] / err
    
        bte_tmp[count].extend(bte)
        count+=1
    
bte = np.mean(bte_tmp, axis = 0)

#%%
sns.set()

fontsize=22
ticksize=20

#fig.suptitle('ntrees = '+str(ntrees),fontsize=25)
plt.plot(range(len(bte)), bte, c='red', marker='.', markersize=14, linewidth=3, linestyle='dashed')
plt.hlines(1, 0, 10, colors='grey', linestyles='dashed',linewidth=1.5)
plt.tick_params(labelsize=ticksize)
plt.xlabel('Number of tasks seen', fontsize=fontsize)
plt.ylabel('BTE Task 1', fontsize=fontsize)

plt.savefig('./result/figs/fig_trees'+str(ntrees)+"__"+model+'.pdf',dpi=300)

# %%
