#%%
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
import pandas as pd
from itertools import product
import seaborn as sns

#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_fte_bte(err, single_err, ntrees):
    bte = [[] for i in range(10)]
    te = [[] for i in range(10)]
    fte = []
    
    for i in range(10):
        for j in range(i,10):
            #print(err[j][i],j,i)
            bte[i].append(err[i][i]/err[j][i])
            te[i].append(single_err[i]/err[j][i])
                
    for i in range(10):
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

def get_error_matrix(filename):
    multitask_df, single_task_df = unpickle(filename)

    err = [[] for _ in range(10)]

    for ii in range(10):
        err[ii].extend(
            1 - np.array(
                multitask_df[multitask_df['base_task']==ii+1]['accuracy']
                )
            )
    single_err = 1 - np.array(single_task_df['accuracy'])

    return single_err, err

#%%
### MAIN HYPERPARAMS ###
ntrees = 10
slots = 10
task_num = 10
shifts = 6
total_alg = 9
alg_name = ['L2N','L2F','L2F(minus)','Prog-NN', 'DF-CNN','EWC','O-EWC','SI','LwF']
model_file = ['dnn0','fixed_uf10','uf10','Prog_NN','DF_CNN','EWC', 'Online_EWC', 'SI', 'LwF']
btes = [[] for i in range(total_alg)]
ftes = [[] for i in range(total_alg)]
tes = [[] for i in range(total_alg)]
########################

#%% code for L2F and L2N
reps = slots*shifts

for alg in range(total_alg): 
    count = 0 
    te_tmp = [[] for _ in range(reps)]
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 

    for slot in range(slots):
        for shift in range(shifts):
            if alg < 3:
                filename = 'result/result/'+model_file[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            else:
                filename = 'benchmarking_algorthms_result/'+model_file[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'

            multitask_df, single_task_df = unpickle(filename)

            single_err, err = get_error_matrix(filename)
            fte, bte, te = get_fte_bte(err,single_err,ntrees)
            
            te_tmp[count].extend(te)
            bte_tmp[count].extend(bte)
            fte_tmp[count].extend(fte)
            count+=1
    
    tes[alg].extend(calc_mean_te(te_tmp,reps=reps))
    btes[alg].extend(calc_mean_bte(bte_tmp,reps=reps))
    ftes[alg].extend(calc_mean_fte(fte_tmp,reps=reps))

#%%
te = {'L2N':np.zeros(10,dtype=float), 'L2F':np.zeros(10,dtype=float),'L2Fc':np.zeros(10,dtype=float), 'Prog-NN':np.zeros(10,dtype=float), 'DF-CNN':np.zeros(10,dtype=float),'EWC':np.zeros(10,dtype=float), 'Online EWC':np.zeros(10,dtype=float), 'SI':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(10):
        te[name][i] = tes[count][i][9-i]

df = pd.DataFrame.from_dict(te)
df = pd.melt(df,var_name='Algorithms', value_name='Transfer Efficieny')

mean_te = {'L2N':[np.mean(te['L2N'])],'L2F':[np.mean(te['L2F'])], 'L2Fc':[np.mean(te['L2Fc'])],
            'Prog-NN':[np.mean(te['Prog-NN'])], 
           'DF-CNN':[np.mean(te['DF-CNN'])],'EWC':[np.mean(te['EWC'])], 
           'Online EWC':[np.mean(te['Online EWC'])], 'SI':[np.mean(te['SI'])], 
           'LwF':[np.mean(te['LwF'])]}
mean_df = pd.DataFrame.from_dict(mean_te)
mean_df = pd.melt(mean_df,var_name='Algorithms', value_name='Transfer Efficieny')

#%%
clr = ["#00008B", "#e41a1c", "#e41a1c", "#a65628", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#CCCC00"]
c = sns.color_palette(clr, n_colors=len(clr))

fontsize=24
ticksize=20

fig, ax = plt.subplots(2,2, figsize=(16,11.5))
# plt.subplots_adjust(right=0.5)
for i, fte in enumerate(ftes):
    if i == 0:
        ax[0][0].plot(np.arange(1,11), fte, c=clr[i], marker='.', markersize=12, label=alg_name[i], linewidth=3)
        continue

    if i == 1:
        ax[0][0].plot(np.arange(1,11), fte, c=clr[i], marker='.', markersize=12, label=alg_name[i], linewidth=3)
        continue

    if i == 2:
        ax[0][0].plot(np.arange(1,11), fte, c=clr[i], marker='.', linestyle='dashed', markersize=12, label=alg_name[i], linewidth=3)
        continue
    
    ax[0][0].plot(np.arange(1,11), fte, c=clr[i], marker='.', markersize=12, label=alg_name[i])
    
ax[0][0].set_xticks(np.arange(1,11))
ax[0][0].set_yticks([0.9, 1, 1.1, 1.2, 1.3,1.4])
ax[0][0].set_ylim(0.85, 1.41)
ax[0][0].tick_params(labelsize=ticksize)
# ax[0].legend(algos, loc='upper left', fontsize=14)
# ax[0].legend(algos, bbox_to_anchor=(1.2, -.2), loc=2, borderaxespad=0)

ax[0][0].set_ylabel('Forward Transfer Efficiency', fontsize=fontsize)
ax[0][0].set_xlabel('Number of tasks seen', fontsize=fontsize)

#ax[0][0].grid(axis='x')

for i in range(task_num - 1):

    et = np.zeros((total_alg,task_num-i))

    for j in range(0,total_alg):
        et[j,:] = np.asarray(btes[j][i])

    ns = np.arange(i + 1, task_num + 1)
    for j in range(0,total_alg):
        if j == 0:
            if i == 0:
                ax[0][1].plot(ns, et[j,:], marker='.', markersize=8, label = alg_name[j], c=c[j], linewidth = 3)
            else:
                ax[0][1].plot(ns, et[j,:], marker='.', markersize=8, c=c[j], linewidth = 3)
        elif j == 1:
            if i == 0:
                ax[0][1].plot(ns, et[j,:], marker='.', markersize=8, label = alg_name[j], c=c[j], linewidth = 3)
            else:
                ax[0][1].plot(ns, et[j,:], marker='.', markersize=8, c=c[j], linewidth = 3)
        elif j==2:
            if i == 0:
                ax[0][1].plot(ns, et[j,:], marker='.', markersize=8, label = alg_name[j], c=c[j], linestyle='dashed', linewidth = 3)
            else:
                ax[0][1].plot(ns, et[j,:], marker='.', markersize=8, c=c[j], linestyle='dashed', linewidth = 3)
        else:
            if i == 0:
                ax[0][1].plot(ns, et[j,:], marker='.', markersize=8, label = alg_name[j], c=c[j])
            else:
                ax[0][1].plot(ns, et[j,:], marker='.', markersize=8, c=c[j])


# ax[1].set_title(ttle, fontsize=20)
ax[0][1].set_xlabel('Number of tasks seen', fontsize=fontsize)
ax[0][1].set_ylabel('Backward Transfer Efficiency', fontsize=fontsize)
# ax.set_ylim(0.05 - 0.01, 0.5 + 0.01)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax[1].legend(loc='upper left', fontsize=12)
ax[0][1].legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=22)
ax[0][1].set_yticks([.4,.6,.8,.9,1, 1.1,1.2])
ax[0][1].set_xticks(np.arange(1,11))
ax[0][1].set_ylim(0.85, 1.19)
ax[0][1].tick_params(labelsize=ticksize)
#ax[0][1].grid(axis='x')


for i in range(task_num- 1):

    et = np.zeros((total_alg,task_num-i))

    for j in range(0,total_alg):
        et[j,:] = np.asarray(tes[j][i])

    ns = np.arange(i + 1, task_num + 1)
    for j in range(0,total_alg):
        if j == 0:
            if i == 0 or i==1:
                ax[1][0].plot(ns, et[j,:], marker='.', markersize=8, label = alg_name[j], c=c[j], linewidth = 3)
            else:
                ax[1][0].plot(ns, et[j,:], marker='.', markersize=8, c=c[j], linewidth = 3)
        elif j == 1:
            if i == 0 or i==1:
                ax[1][0].plot(ns, et[j,:], marker='.', markersize=8, label = alg_name[j], c=c[j], linewidth = 3)
            else:
                ax[1][0].plot(ns, et[j,:], marker='.', markersize=8, c=c[j], linewidth = 3)        
        elif j == 2:
            if i == 0 or i==1:
                ax[1][0].plot(ns, et[j,:], marker='.', markersize=8, label = alg_name[j], c=c[j], linestyle='dashed', linewidth = 3)
            else:
                ax[1][0].plot(ns, et[j,:], marker='.', markersize=8, c=c[j], linestyle='dashed', linewidth = 3)
        else:
            if i == 0 or i==1:
                ax[1][0].plot(ns, et[j,:], marker='.', markersize=8, label = alg_name[j], c=c[j])
            else:
                ax[1][0].plot(ns, et[j,:], marker='.', markersize=8, c=c[j])


# ax[1].set_title(ttle, fontsize=20)
ax[1][0].set_xlabel('Number of tasks seen', fontsize=fontsize)
ax[1][0].set_ylabel('Transfer Efficiency', fontsize=fontsize)
# ax.set_ylim(0.05 - 0.01, 0.5 + 0.01)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax[1].legend(loc='upper left', fontsize=12)
#ax[1][0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=22)
ax[1][0].set_yticks([.4,.6,.8,.9,1, 1.1,1.2])
ax[1][0].set_xticks(np.arange(1,11))
ax[1][0].set_ylim(0.85, 1.26)
ax[1][0].tick_params(labelsize=ticksize)
#ax[1][0].grid(axis='x')

ax[0][0].hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)
ax[0][1].hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)
ax[1][0].hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)



ax[1][1].tick_params(labelsize=22)
ax_ = sns.stripplot(x="Algorithms", y="Transfer Efficieny", data=df, palette=c, size=6, ax=ax[1][1])
ax[1][1].hlines(1, -1,8, colors='grey', linestyles='dashed',linewidth=1.5)
sns.boxplot(x="Algorithms", y="Transfer Efficieny", data=mean_df, palette=c, linewidth=3, ax=ax[1][1])
ax_.set_yticks([.4,.6,.8,1, 1.2,1.4])
ax_.set_xlabel('Algorithms', fontsize=fontsize)
ax_.set_ylabel('Final Transfer Efficiency', fontsize=fontsize)
ax_.set_xticklabels(
    ['L2N','L2F','L2F(minus)','Prog-NN','DF-CNN','EWC','O-EWC','SI','LwF'],
    fontsize=12,rotation=45,ha="right",rotation_mode='anchor'
    )

right_side = ax[0][0].spines["right"]
right_side.set_visible(False)
top_side = ax[0][0].spines["top"]
top_side.set_visible(False)

right_side = ax[0][1].spines["right"]
right_side.set_visible(False)
top_side = ax[0][1].spines["top"]
top_side.set_visible(False)

right_side = ax[1][0].spines["right"]
right_side.set_visible(False)
top_side = ax[1][0].spines["top"]
top_side.set_visible(False)

right_side = ax[1][1].spines["right"]
right_side.set_visible(False)
top_side = ax[1][1].spines["top"]
top_side.set_visible(False)

plt.tight_layout()
# lgd = fig.legend(algos, bbox_to_anchor=(1, 0.45), loc='center left', fontsize=18)
plt.savefig('result/figs/benchmark.pdf', dpi=500)

#%%
