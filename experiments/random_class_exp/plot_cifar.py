#%%
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
from itertools import product

### MAIN HYPERPARAMS ###
slots = 10
shifts = 6
alg_num = 1
task_num = 10
model_ntree_pairs = [("uf", 10), ("dnn", 0)]
########################

ax = plt.subplot(111)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

for pair in model_ntree_pairs:
    model = pair[0]
    ntrees = pair[1]
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
    fontsize=22
    ticksize=20

    #fig.suptitle('ntrees = '+str(ntrees),fontsize=25)
    ax.plot(range(len(bte)), bte, c="#00008B" if model == "dnn" else "#e41a1c", linewidth=3, linestyle="solid", label = "L2N" if model == "dnn" else "L2F")

ax.set_xlabel('Number of Tasks Seen', fontsize=fontsize)
ax.set_ylabel('Transfer Efficiency (Task 1)', fontsize=fontsize)
ax.set_yticks([1.0, 1.1, 1.2])
ax.tick_params(labelsize=ticksize)
ax.legend(fontsize=22)

plt.tight_layout()
plt.savefig('./result/figs/fig_trees'+str(ntrees)+"__"+model+'.pdf',dpi=300)

# %%
