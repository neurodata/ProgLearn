#%%
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
# %%
alg_name = ['PLN','PLF','Prog_NN', 'DF_CNN','LwF','EWC','O-EWC','SI', 'Replay \n (increasing amount)', 'Replay \n (fixed amount)', 'None']
model_file = ['dnn','uf','Prog_NN','DF_CNN', 'LwF', 'EWC', 'OEWC', 'SI', 'offline', 'exact', 'None']
total_alg = 11
slots = 10
shifts = 6
time_info = [[] for i in range(total_alg)]
mem_info = [[] for i in range(total_alg)]

for alg in range(total_alg): 
    if alg < 2:
        filename = './result/time_res/'+model_file[alg]+'pickle'
    elif alg == 2 or alg == 3:
        filename = './result/time_res/'+model_file[alg]+str(1)+'_'+str(1)+'.pickle'
    else:
        filename = './result/time_res/'+model_file[alg]+'-'+str(1)+'-'+str(1)+'.pkl'

    with open(filename,'rb') as f:
            data = pickle.load(f)

    time_info[alg].extend(data)
# %%
for alg in range(total_alg): 
    if alg < 2:
        filename = './result/mem_res/'+model_file[alg]+'pickle'
    elif alg == 2 or alg == 3:
        filename = './result/mem_res/'+model_file[alg]+str(1)+'_'+str(1)+'.pickle'
    else:
        filename = './result/mem_res/'+model_file[alg]+'-'+str(1)+'-'+str(1)+'.pkl'

    with open(filename,'rb') as f:
            data = pickle.load(f)

    mem_info[alg].extend(data)
# %%
fontsize=20
ticksize=18
fig = plt.figure(constrained_layout=True,figsize=(14,5))
#fig, ax = plt.subplots(1,2, figsize=(14,5))
gs = fig.add_gridspec(5, 14)

clr = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928"]
c = sns.color_palette(clr, n_colors=len(clr))
marker_style = ['.', '.', '.', '.', '.', '+', 'o', '*', '.', '+', 'o']

ax = fig.add_subplot(gs[:5,:5])
for alg_no,alg in enumerate(alg_name):
    if alg_no<2:
        ax.plot(np.arange(1,11),time_info[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3, marker=marker_style[alg_no])
    else:
        ax.plot(np.arange(1,11),time_info[alg_no], c=c[alg_no], label=alg_name[alg_no], marker=marker_style[alg_no])

#ax[0].set_yticks([.8,.9,1,1.1,1.2])
#ax[0].set_ylim([0.91,1.17])
ax.set_xticks(np.arange(1,11))
ax.tick_params(labelsize=ticksize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('Time', fontsize=fontsize)
#ax[0].set_title("Label Shuffled CIFAR", fontsize = fontsize)
#ax.hlines(1,1,10, colors='grey', linestyles='dashed',linewidth=1.5)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
plt.tight_layout()

ax = fig.add_subplot(gs[:5,6:11])
for alg_no,alg in enumerate(alg_name):
    if alg_no<2:
        ax.plot(np.arange(1,11),mem_info[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3, marker=marker_style[alg_no])
    else:
        ax.plot(np.arange(1,11),mem_info[alg_no], c=c[alg_no], label=alg_name[alg_no], marker=marker_style[alg_no])

#ax[0].set_yticks([.8,.9,1,1.1,1.2])
#ax[0].set_ylim([0.91,1.17])
ax.set_xticks(np.arange(1,11))
ax.tick_params(labelsize=ticksize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('Memory', fontsize=fontsize)
#ax[1].set_title("Label Shuffled CIFAR", fontsize = fontsize)
#ax.hlines(1,1,10, colors='grey', linestyles='dashed',linewidth=1.5)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
handles, labels_ = ax.get_legend_handles_labels()
ax.legend(handles, labels_, bbox_to_anchor=(1, 1), fontsize=14, frameon=False)
plt.tight_layout()

plt.savefig('./result/figs/scaling.pdf')
# %%
