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


#%%
reps = slots *shifts

btes = []

for slot in range(0, slots):
    for shift in range(shifts):
        filename = 'result/'+model+str(ntrees)+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
        df = unpickle(filename)

        err = 1 - df['accuracy']
        bte = err[0] / err
    
        btes.append(bte)

    
bte = np.mean(btes, axis = 0)

#%%
sns.set()

fontsize=22
ticksize=20

#fig.suptitle('ntrees = '+str(ntrees),fontsize=25)
plt.plot(range(len(bte)), bte, c='red', marker='.', markersize=14, linewidth=3, linestyle="dashed" if model == "dnn" else "solid", label = "L2N" if model == "dnn" else "L2F")
plt.hlines(1, 0, len(bte), colors='grey', linestyles="dashed",linewidth=1.5)
plt.tick_params(labelsize=ticksize)
plt.xlabel('Number of tasks seen', fontsize=fontsize)
plt.ylabel('Transfer Efficiency (Task 1)', fontsize=fontsize)
plt.legend()

plt.savefig('./result/figs/fig_trees'+str(ntrees)+"__"+model+'.pdf',dpi=300)

# %%
