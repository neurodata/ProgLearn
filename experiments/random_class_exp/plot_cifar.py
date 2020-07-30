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
task_num = 9
model = 'uf'
ntrees = 10
########################


#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


#%%
reps = slots *shifts

btes = []
train_times = []
inference_times = []

for slot in range(0, slots):
    for shift in range(shifts):
        filename = 'result/'+model+str(ntrees)+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
        df = unpickle(filename)

        err = 1 - df['accuracy']
        bte = err[0] / err

        btes.append(bte)
        train_times.append(df['train_times'])
        inference_times.append(df['inference_times'])


bte = np.mean(btes, axis = 0)
train_time_across_tasks = np.mean(train_times, axis = 0)
inference_time_across_tasks = np.mean(inference_times, axis = 0)

#%%
fontsize=22
ticksize=20

ax = plt.subplot(111)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

#fig.suptitle('ntrees = '+str(ntrees),fontsize=25)
ax.plot(range(len(bte)), bte, c="blue" if model == "dnn" else "red", linewidth=3, linestyle="solid", label = "L2N" if model == "dnn" else "L2F")

ax.set_xlabel('Number of Tasks Seen', fontsize=fontsize)
ax.set_ylabel('Transfer Efficiency (Task 1)', fontsize=fontsize)
ax.set_yticks([1.0, 1.1, 1.2])
ax.tick_params(labelsize=ticksize)
ax.legend(fontsize=22)

plt.tight_layout()
plt.savefig('./result/figs/random_class.pdf',dpi=300)
plt.close()

ax = plt.subplot(111)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

#fig.suptitle('ntrees = '+str(ntrees),fontsize=25)
ax.plot(range(len(train_time_across_tasks)), train_time_across_tasks, linewidth=3, linestyle="solid", label = "Train Time")
ax.plot(range(len(inference_time_across_tasks)), inference_time_across_tasks, linewidth=3, linestyle="solid", label = "Inference Time")

ax.set_xlabel('Number of Tasks Seen', fontsize=fontsize)
ax.set_ylabel('Time (ms)', fontsize=fontsize)
ax.tick_params(labelsize=ticksize)
ax.legend(fontsize=22)

plt.tight_layout()
plt.savefig('./result/figs/random_class_time.pdf',dpi=300)

# %%
