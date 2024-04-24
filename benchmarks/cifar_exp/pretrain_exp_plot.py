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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap
#%%
def register_palette(name, clr):
    # relative positions of colors in cmap/palette 
    pos = [0.0,1.0]

    colors=['#FFFFFF', clr]
    cmap = LinearSegmentedColormap.from_list("", list(zip(pos, colors)))
    register_cmap(name, cmap)

def calc_forget(err, reps, total_task=10):
#Tom Vient et al
    forget = 0
    for ii in range(total_task-1):
        forget += err[ii][ii] - err[total_task-1][ii]

    forget /= (total_task-1)
    return forget/reps

def calc_transfer(err, single_err, reps, total_task=10):
#Tom Vient et al
    transfer = np.zeros(total_task,dtype=float)

    for ii in range(total_task):
        transfer[ii] = (single_err[ii] - err[total_task-1][ii])/reps

    return np.mean(transfer)

def calc_acc(err, reps, total_task=10):
#Tom Vient et al
    acc = 0
    for ii in range(total_task):
        acc += (1-err[total_task-1][ii]/reps)
    return acc/total_task

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def calc_final_acc(err, reps):
    avg_acc = []
    for err_ in err[-1][::-1]:
        avg_acc.append(1-err_/reps)
    
    return avg_acc

def calc_avg_acc(err, reps):
    avg_acc = np.zeros(10, dtype=float)
    avg_var = np.zeros(10, dtype=float)
    for i in range(10):
        avg_acc[i] = (1*(i+1) - np.sum(err[i])/reps + (9-i)*.1)/10
        avg_var[i] = np.var(1-np.array(err[i])/reps)
    return avg_acc, avg_var

def calc_avg_single_acc(err, reps):
    avg_acc = np.zeros(10, dtype=float)
    avg_var = np.zeros(10, dtype=float)
    for i in range(10):
        avg_acc[i] = (1*(i+1) - np.sum(err[:i+1])/reps + (9-i)*.1)/10
        avg_var[i] = np.var(1-np.array(err[:i+1])/reps)
    return avg_acc, avg_var

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

def sum_error_matrix(error_mat1, error_mat2):
    err = [[] for _ in range(10)]

    for ii in range(10):
        err[ii].extend(
            list(
                np.asarray(error_mat1[ii]) +
                np.asarray(error_mat2[ii])
            )
        )
    return err

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
                c='k'
                )
            
#%%
### MAIN HYPERPARAMS ###
ntrees = 10
slots = 10
task_num = 10
shifts = 1
total_alg = 2
alg_name = ['SiLLy-N', 'SiLLy-N (pretrained)']
model_file = ['dnn_not_pretrained', 'dnn']
btes = [[] for i in range(total_alg)]
ftes = [[] for i in range(total_alg)]
tes = [[] for i in range(total_alg)]
acc = [[] for i in range(total_alg)]

########################
# %%
reps = slots*shifts

for alg in range(total_alg): 
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for slot in range(slots):
        for shift in range(shifts):
            filename = '/Users/jayantadey/ProgLearn/benchmarks/cifar_exp/result_pretrained/'+model_file[alg]+'_'+str(slot)+'_'+str(shift)+'.pickle'
           
            multitask_df, single_task_df = unpickle(filename)

            single_err_, err_ = get_error_matrix(filename)
            
            if count == 0:
                single_err, err = single_err_, err_
            else:
                err = sum_error_matrix(err, err_)
                single_err = list(
                    np.asarray(single_err) + np.asarray(single_err_)
                )

            count += 1
    #single_err /= reps
    #err /= reps
    fte, bte, te = get_fte_bte(err,single_err)
    avg_acc = calc_final_acc(err, reps)

    btes[alg].extend(bte)
    ftes[alg].extend(fte)
    tes[alg].extend(te)
    acc[alg].extend(avg_acc)
#%%
fte_end = {'SiLLy-N':np.zeros(10,dtype=float), 'SiLLy-N (pretrained)':np.zeros(10,dtype=float)
          }

task_order = []
t = 1
for count,name in enumerate(fte_end.keys()):
    #print(name, count)
    for i in range(10):
        fte_end[name][9-i] = np.log(ftes[count][i])
        task_order.append(t)
        t += 1

tmp_fle = {}
for id in fte_end.keys():
    tmp_fle[id] = fte_end[id]

df_fle = pd.DataFrame.from_dict(tmp_fle)
df_fle = pd.melt(df_fle,var_name='Algorithms', value_name='Forward Transfer Efficieny')
df_fle.insert(2, "Task ID", task_order)
#%%
bte_end = {'SiLLy-N':np.zeros(10,dtype=float), 'SiLLy-N (pretrained)':np.zeros(10,dtype=float)
          }


for count,name in enumerate(bte_end.keys()):
    #print(name, count)
    for i in range(10):
        bte_end[name][9-i] = np.log(btes[count][i][9-i])

tmp_ble = {}
for id in alg_name:
    tmp_ble[id] = bte_end[id]

df_ble = pd.DataFrame.from_dict(tmp_ble)
df_ble = pd.melt(df_ble,var_name='Algorithms', value_name='Backward Transfer Efficieny')
df_ble.insert(2, "Task ID", task_order)

#%%
te_end = {'SiLLy-N':np.zeros(10,dtype=float), 'SiLLy-N (pretrained)':np.zeros(10,dtype=float)
          }


for count,name in enumerate(bte_end.keys()):
    #print(name, count)
    for i in range(10):
        te_end[name][9-i] = np.log(tes[count][i][9-i])

tmp_le = {}
for id in alg_name:
    tmp_le[id] = te_end[id]

df_le = pd.DataFrame.from_dict(tmp_le)
df_le = pd.melt(df_le,var_name='Algorithms', value_name='Transfer Efficiency')
df_le.insert(2, "Task ID", task_order)
#%%
acc_end = {'SiLLy-N':np.zeros(10,dtype=float), 'SiLLy-N (pretrained)':np.zeros(10,dtype=float)
          }


for count,name in enumerate(bte_end.keys()):
    #print(name, count)
    for i in range(10):
        acc_end[name][i] = acc[count][i]

tmp_acc = {}
for id in alg_name:
    tmp_acc[id] = acc_end[id]

df_acc = pd.DataFrame.from_dict(tmp_acc)
df_acc = pd.melt(df_acc,var_name='Algorithms', value_name='Accuracy')
df_acc.insert(2, "Task ID", task_order)
#%%
universal_clr_dict = {'SiLLy-N': 'b',
                      'SiLLy-N (pretrained)': 'r',
                    }
for ii, name in enumerate(universal_clr_dict.keys()):
    print(name)
    register_palette(name, universal_clr_dict[name])
    
# %%
fig = plt.figure(constrained_layout=True,figsize=(46,16))
gs = fig.add_gridspec(16,46)

names = ['SiLLy-N (not\npretrained)', 'SiLLy-N     \n (pretrained)']
#c_top = sns.color_palette('Reds', n_colors=10)
c_top = []
for name in alg_name:
    c_top.extend(
        sns.color_palette(
            name, 
            n_colors=task_num
            )
        )


fontsize=40
ticksize=38
legendsize=16


ax = fig.add_subplot(gs[2:10,10:18])

ax_ = sns.stripplot(x='Algorithms', y='Forward Transfer Efficieny', hue='Task ID', data=df_fle, palette=c_top, ax=ax, size=25, legend=None, jitter=.05)
ax_.set_xticklabels(
    names,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
for xtick, color in zip(ax_.get_xticklabels(), alg_name):
        xtick.set_color(universal_clr_dict[color])

#ax.set_title('Resource Constrained FL', fontsize=fontsize)

ax_.set_yticks([-.3,0,.2])
ax.set_ylabel('Forward Transfer', fontsize=fontsize)
ax.set_xlabel('', fontsize=fontsize)
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.hlines(0, 0, 1, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

handles_task, labels_task = ax.get_legend_handles_labels()

#########################################################
#ax[0][0].grid(axis='x')
ax = fig.add_subplot(gs[2:10,20:28])

ax_ = sns.stripplot(x='Algorithms', y='Backward Transfer Efficieny', hue='Task ID', data=df_ble, palette=c_top, ax=ax, size=25, legend=None, jitter=.05)
ax_.set_xticklabels(
    names,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
for xtick, color in zip(ax_.get_xticklabels(), alg_name):
        xtick.set_color(universal_clr_dict[color])

ax_.set_yticks([-.4,0,.1])
#ax.set_title('Resource Constrained BL', fontsize=fontsize)
ax.set_ylabel('Backward Transfer', fontsize=fontsize)
ax.hlines(0, 0, 1, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')
ax.set_xlabel('')
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
########################################################
ax = fig.add_subplot(gs[2:10,:8])

ax_ = sns.stripplot(x='Algorithms', y='Transfer Efficiency', hue='Task ID', data=df_le, palette=c_top, ax=ax, size=25, legend=None, jitter=.05)
ax_.set_xticklabels(
    names,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
for xtick, color in zip(ax_.get_xticklabels(), alg_name):
        xtick.set_color(universal_clr_dict[color])

ax_.set_yticks([0,.2])
#ax.set_title('Resource Constrained BL', fontsize=fontsize)
ax.set_ylabel('Transfer', fontsize=fontsize)
ax.hlines(0, 0, 1, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')
ax.set_xlabel('')
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)


##########################################################
ax = fig.add_subplot(gs[2:10,30:38])

ax_ = sns.stripplot(x='Algorithms', y='Accuracy', hue='Task ID', data=df_acc, palette=c_top, ax=ax, size=25, legend=None, jitter=.05)
ax_.set_xticklabels(
    names,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
for xtick, color in zip(ax_.get_xticklabels(), alg_name):
        xtick.set_color(universal_clr_dict[color])

ax_.set_yticks([0,.5])
#ax.set_title('Resource Constrained BL', fontsize=fontsize)
ax.set_ylabel('Accuracy', fontsize=fontsize)
ax.set_xlabel('')
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

fig.text(.3,.95,'CIFAR 10X10 (ResNet 50 Encoder)', fontsize=55)
#########################################################
plt.savefig('/Users/jayantadey/ProgLearn/benchmarks/cifar_exp/result_pretrained/cifar_pretrained.pdf', bbox_inches='tight')

# %%
