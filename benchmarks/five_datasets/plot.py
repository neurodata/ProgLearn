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
#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def calc_avg_acc(err, reps):
    avg_acc = np.zeros(5, dtype=float)
    avg_var = np.zeros(5, dtype=float)
    for i in range(5):
        avg_acc[i] = (1*(i+1) - np.sum(err[i])/reps + (4-i)*.1)/10
        avg_var[i] = np.var(1-np.array(err[i])/reps)
    return avg_acc, avg_var

def calc_avg_single_acc(err, reps):
    avg_acc = np.zeros(5, dtype=float)
    avg_var = np.zeros(5, dtype=float)
    for i in range(5):
        avg_acc[i] = (1*(i+1) - np.sum(err[:i+1])/reps + (4-i)*.1)/10
        avg_var[i] = np.var(1-np.array(err[:i+1])/reps)
    return avg_acc, avg_var
    
def get_fte_bte(err, single_err):
    bte = [[] for i in range(5)]
    te = [[] for i in range(5)]
    fte = []
    
    for i in range(5):
        for j in range(i,5):
            #print(err[j][i],j,i)
            bte[i].append(err[i][i]/err[j][i])
            te[i].append(single_err[i]/err[j][i])
                
    for i in range(5):
        fte.append(single_err[i]/err[i][i])
            
            
    return fte,bte,te

def calc_mean_bte(btes,task_num=6,reps=10):
    mean_bte = [[] for i in range(task_num)]


    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(btes[i][j])
        
        tmp=tmp/reps
        mean_bte[j].extend(tmp)
            
    return mean_bte     

def calc_mean_te(tes,task_num=6,reps=10):
    mean_te = [[] for i in range(task_num)]

    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(tes[i][j])
        
        tmp=tmp/reps
        mean_te[j].extend(tmp)
                                             
    return mean_te 

def calc_mean_fte(ftes,task_num=5,reps=1):
    fte = np.asarray(ftes)
    
    return list(np.mean(np.asarray(fte),axis=0))

def get_error_matrix(filename):
    multitask_df, single_task_df = unpickle(filename)

    err = [[] for _ in range(5)]

    for ii in range(5):
        err[ii].extend(
            1 - np.array(
                multitask_df[multitask_df['base_task']==ii+1]['accuracy']
                )
            )
    single_err = 1 - np.array(single_task_df['accuracy'])

    return single_err, err

def sum_error_matrix(error_mat1, error_mat2):
    err = [[] for _ in range(6)]

    for ii in range(6):
        err[ii].extend(
            list(
                np.asarray(error_mat1[ii]) +
                np.asarray(error_mat2[ii])
            )
        )
    return err

def stratified_scatter(te_dict,axis_handle,s,color,style):
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
                s=s,
                c='k',
                marker=style[algo_no]
                )

#%%
### MAIN HYPERPARAMS ###
task_num = 5
total_alg = 13
combined_alg_name = ['SynN','SynF', 'Model Zoo', 'LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay', 'Partial Replay', 'None']
btes = [[] for i in range(total_alg)]
ftes = [[] for i in range(total_alg)]
tes = [[] for i in range(total_alg)]
model_file_combined = ['synn','synf', 'model_zoo', 'LwF', 'EWC', 'OEWC', 'si', 'er', 'agem', 'tag', 'offline', 'exact', 'None']
avg_acc = [[] for i in range(total_alg)]
avg_var = [[] for i in range(total_alg)]
avg_single_acc = [[] for i in range(total_alg)]
avg_single_var = [[] for i in range(total_alg)]
########################

#%% code for 500 samples
reps = 1

for alg in range(total_alg): 
    count = 0 

    for rep in range(reps):
        filename = './results/'+model_file_combined[alg]+'.pickle'

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
    avg_acc_, avg_var_ = calc_avg_acc(err, reps)
    avg_single_acc_, avg_single_var_ = calc_avg_single_acc(single_err, reps)

    btes[alg].extend(bte)
    ftes[alg].extend(fte)
    tes[alg].extend(te)
    avg_acc[alg]= avg_acc_
    avg_var[alg] = avg_var_
    avg_single_acc[alg]= avg_single_acc_
    avg_single_var[alg] = avg_single_var_

#%%
te = {'SynN':np.zeros(5,dtype=float), 'SynF':np.zeros(5,dtype=float), 'model_zoo':np.zeros(5,dtype=float), 
    'LwF':np.zeros(5,dtype=float), 'EWC':np.zeros(5,dtype=float), 
    'O-EWC':np.zeros(5,dtype=float), 'SI':np.zeros(5,dtype=float),
    'ER':np.zeros(5,dtype=float), 'A-GEM':np.zeros(5,dtype=float),
    'TAG':np.zeros(5,dtype=float),
    'Total Replay':np.zeros(5,dtype=float), 'Partial Replay':np.zeros(5,dtype=float), 
    'None':np.zeros(5,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(5):
        te[name][i] = np.log(tes[count][i][4-i])


df = pd.DataFrame.from_dict(te)
df = pd.melt(df,var_name='Algorithms', value_name='Transfer Efficieny')

# %%
fig = plt.figure(constrained_layout=True,figsize=(28,17))
gs = fig.add_gridspec(17, 28)

marker_style = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']
marker_style_scatter = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']

clr_combined = ["#377eb8", "#e41a1c", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928"]
c_combined = sns.color_palette(clr_combined, n_colors=total_alg)

fontsize=29
ticksize=26
legendsize=14

ax = fig.add_subplot(gs[:7,:7])

for i, fte in enumerate(ftes):
    fte[0] = 1
    if i == 0:
        ax.plot(np.arange(1,6), fte, color=c_combined[i], marker=marker_style[i], markersize=12, label=combined_alg_name[i], linewidth=3)
        continue

    if i == 1:
        ax.plot(np.arange(1,6), fte, color=c_combined[i], marker=marker_style[i], markersize=12, label=combined_alg_name[i], linewidth=3)
        continue
    
    ax.plot(np.arange(1,6), fte, color=c_combined[i], marker=marker_style[i], markersize=12, label=combined_alg_name[i])
    
ax.set_xticks(np.arange(1,6))
ax.set_yticks([0.5, 1, 2, 3])
#ax.set_yticks([])
#ax.text(0, np.mean(ax.get_ylim()), "%s" % str(0), fontsize=26)
#ax.yaxis.set_major_locator(plt.LogLocator(subs=(0.9, 1, 1.1, 1.2, 1.3)))
#ax.set_ylim(0.89, 1.31)

log_lbl = np.round(
    np.log([0.5,1,2,3]),
    1
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)

ax.tick_params(labelsize=ticksize)

ax.set_ylabel('log Forward LE', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,6, colors='grey', linestyles='dashed',linewidth=1.5)



#ax[0][0].grid(axis='x')
ax = fig.add_subplot(gs[:7,9:16])

for i in range(task_num - 1):

    et = np.zeros((total_alg,task_num-i))

    for j in range(0,total_alg):
        et[j,:] = np.asarray(btes[j][i])

    ns = np.arange(i + 1, task_num + 1)
    for j in range(0,total_alg):
        if j == 0:
            if i == 0:
                ax.plot(ns, et[j,:], marker=marker_style[j], markersize=8, label = combined_alg_name[j], color=c_combined[j], linewidth = 3)
            else:
                ax.plot(ns, et[j,:], marker=marker_style[j], markersize=8, color=c_combined[j], linewidth = 3)
        elif j == 1:
            if i == 0:
                ax.plot(ns, et[j,:], marker=marker_style[j], markersize=8, label = combined_alg_name[j], color=c_combined[j], linewidth = 3)
            else:
                ax.plot(ns, et[j,:], marker=marker_style[j], markersize=8, color=c_combined[j], linewidth = 3)
        else:
            if i == 0:
                ax.plot(ns, et[j,:], marker=marker_style[j], markersize=8, label = combined_alg_name[j], color=c_combined[j])
            else:
                ax.plot(ns, et[j,:], marker=marker_style[j], markersize=8, color=c_combined[j])

ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('log Backward LE', fontsize=fontsize)

ax.set_xticks(np.arange(1,6))
ax.set_yticks([.2,1,2,3])
#ax.set_xticks(np.arange(1,11))
#ax.set_ylim(0.76, 1.25)

log_lbl = np.round(
    np.log([.2,1,2,3]),
    1
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)

ax.tick_params(labelsize=ticksize)
#ax[0][1].grid(axis='x')

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,6, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

handles, labels_ = ax.get_legend_handles_labels()
#ax.legend(loc='center left', bbox_to_anchor=(.8, 0.5), fontsize=legendsize+16)


ax = fig.add_subplot(gs[:7,18:25])
ax.tick_params(labelsize=22)
ax_ = sns.boxplot(
    x="Algorithms", y="Transfer Efficieny", data=df, palette=c_combined, whis=np.inf,
    ax=ax, showfliers=False, notch=1
    )
ax.hlines(0, -1,12, colors='grey', linestyles='dashed',linewidth=1.5)
#sns.boxplot(x="Algorithms", y="Transfer Efficieny", data=mean_df, palette=c, linewidth=3, ax=ax[1][1])
#ax_=sns.pointplot(x="Algorithms", y="Transfer Efficieny", data=df_500, join=False, color='grey', linewidth=1.5, ci='sd',ax=ax)
#ax_.set_yticks([.4,.6,.8,1, 1.2,1.4])
ax_.set_xlabel('', fontsize=fontsize)
ax.set_ylabel('log LE after 5 Tasks', fontsize=fontsize-5)
ax_.set_xticklabels(
    ['SynN','SynF', 'Model zoo', 'LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay','Partial Replay', 'None'],
    fontsize=18,rotation=65,ha="right",rotation_mode='anchor'
    )

stratified_scatter(te,ax,16,c_combined,marker_style_scatter)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

fig.legend(handles, labels_, bbox_to_anchor=(.97, .7), fontsize=legendsize+12, frameon=False)

ax = fig.add_subplot(gs[10:17,4:11])

for i in range(total_alg):
    if i==0 or i==1:
        ax.plot(np.arange(1,6,1) ,avg_acc[i], color=c_combined[i], marker=marker_style[i], linewidth=3)
    else:
        ax.plot(np.arange(1,6,1) ,avg_acc[i], color=c_combined[i], marker=marker_style[i])
    #ax.fill_between(np.arange(1,6,1), avg_acc[i]-1.96*avg_var[i], avg_acc[i]+1.96*avg_var[i], facecolor=c_combined[i], alpha=.3)

ax.hlines(.1, 1,5, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')
ax.set_yticks([.1,.2,.3,.4,.5])
ax.set_xticks(np.arange(1,6))
ax.tick_params(labelsize=ticksize)
ax.set_ylabel('Accuracy[$\pm$ std dev.]', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

############################
ax = fig.add_subplot(gs[10:17,15:22])

for i in range(total_alg):
    if i==0 or i==1:
        ax.plot(np.arange(1,6,1) ,avg_single_acc[i], color=c_combined[i], marker=marker_style[i], linewidth=3)
    else:
        ax.plot(np.arange(1,6,1) ,avg_single_acc[i], color=c_combined[i], marker=marker_style[i])
   # ax.fill_between(np.arange(1,6,1), avg_single_acc[i]-1.96*avg_single_var[i], avg_single_acc[i]+1.96*avg_single_var[i], facecolor=c_combined[i], alpha=.3)

ax.hlines(.1, 1,5, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')
ax.set_yticks([.1,.2,.3,.4,.5])
ax.set_xticks(np.arange(1,6))
ax.tick_params(labelsize=ticksize)
ax.set_ylabel('Single task accuracy', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig('five_dataset.pdf')
# %%