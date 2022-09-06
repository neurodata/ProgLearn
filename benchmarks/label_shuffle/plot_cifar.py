#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_fte_bte(err, single_err):
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
    multitask_df = unpickle(filename)[0]
    err = 1 - np.array(multitask_df[multitask_df['task']==1]['accuracy'])

    return err

def get_error_matrix2(filename):
    multitask_df = unpickle(filename)
    err = 1 - np.array(multitask_df['task_1_accuracy'])

    return err

#%%
alg_name = ['SynN','SynF','Prog_NN', 'DF_CNN', 'Model Zoo', 'LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay', 'Partial Replay', 'None']
model_file = ['dnn0','uf10','Prog_NN','DF_CNN', 'model_zoo', 'LwF', 'EWC', 'OEWC', 'SI', 'er', 'agem', 'tag', 'offline', 'exact', 'None']
total_alg = 15
slots = 10
shifts = 6

#%% claculate TE for label shuffle
reps = slots*shifts
tes_label_shuffle = [[] for i in range(total_alg)]

for alg in range(total_alg): 
    err_ = np.zeros(10,dtype=float)

    for shift in range(shifts):
        if alg < 2:
            filename = './result/'+model_file[alg]+'_'+str(shift+1)+'.pickle'
        elif alg == 2 or alg == 3 or alg ==4:
            filename = './result/'+model_file[alg]+'_'+str(shift+1)+'.pickle'
        else:
            filename = './result/'+model_file[alg]+'-'+str(shift+1)+'.pickle'


        if alg!=2:
            if alg == 3 or alg==9 or alg==10 or alg==11 or alg==4:
                err_ += np.ravel(np.array(get_error_matrix2(filename)))
            else:
                err_ += np.ravel(np.array(get_error_matrix(filename)))
    

    if alg == 2:
        tes_label_shuffle[alg].extend([1]*10)
    else:
        err_ /= reps
        te = err_[0] / err_
        tes_label_shuffle[alg].extend(te)


# %%
fontsize=24
ticksize=22
fig = plt.figure(constrained_layout=True,figsize=(10,8))
gs = fig.add_gridspec(8, 10)

clr = ["#377eb8", "#e41a1c", "#f781bf", "#984ea3", "#984ea3", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928"]
c = sns.color_palette(clr, n_colors=len(clr))
marker_style = ['.', '.', '.', '.', 'v', '.', '.', '.', '+', 'o', '*', '.', '+', 'x', 'v']

ax = fig.add_subplot(gs[:7,:7])

for alg_no,alg in enumerate(alg_name):
    if alg_no<2:
        ax.plot(np.arange(1,11),tes_label_shuffle[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3, marker=marker_style[alg_no])
    else:
        ax.plot(np.arange(1,11),tes_label_shuffle[alg_no], c=c[alg_no], label=alg_name[alg_no], marker=marker_style[alg_no])

ax.set_yticks([.5,.7,.9,1.2])

log_lbl = np.round(
    np.log([.5,.7,.9,1.2]),
    1
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)

#ax[0].set_ylim([0.91,1.17])
ax.set_xticks(np.arange(1,11))
ax.tick_params(labelsize=ticksize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('log Backward LE', fontsize=fontsize)
ax.set_title("Label Shuffled CIFAR", fontsize = fontsize)
ax.hlines(1,1,10, colors='grey', linestyles='dashed',linewidth=1.5)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
fig.legend(bbox_to_anchor=(.99, .93), fontsize=20, frameon=False)
plt.tight_layout()

plt.savefig('./result/figs/adversary_5000.pdf', dpi=500)


# %%
