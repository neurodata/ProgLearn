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
                s=s,
                c='k'
                )

   

#%%
### MAIN HYPERPARAMS ###
ntrees = 10
slots = 10
task_num = 10
shifts = 6
total_alg = 8
alg_name = ['L2N','L2F','Prog-NN', 'DF-CNN','LwF','EWC','O-EWC','SI']
model_file_5000 = ['dnn0','fixed_uf5000_40','Prog_NN','DF_CNN', 'LwF','EWC', 'Online_EWC', 'SI']

btes_5000 = [[] for i in range(total_alg)]
ftes_5000 = [[] for i in range(total_alg)]
tes_5000 = [[] for i in range(total_alg)]
########################

#%% code for 5000 samples
reps = shifts

for alg in range(total_alg): 
    count = 0 
    te_tmp = [[] for _ in range(reps)]
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 

    for shift in range(shifts):
        if alg < 2:
            filename = 'result/result/'+model_file_5000[alg]+'_'+str(shift+1)+'_0'+'.pickle'
        else:
            filename = 'benchmarking_algorthms_result/'+model_file_5000[alg]+'_'+str(shift+1)+'.pickle'

        multitask_df, single_task_df = unpickle(filename)

        single_err, err = get_error_matrix(filename)
        fte, bte, te = get_fte_bte(err,single_err)
            
        te_tmp[count].extend(te)
        bte_tmp[count].extend(bte)
        fte_tmp[count].extend(fte)
        count+=1
    
    tes_5000[alg].extend(calc_mean_te(te_tmp,reps=reps))
    btes_5000[alg].extend(calc_mean_bte(bte_tmp,reps=reps))
    ftes_5000[alg].extend(calc_mean_fte(fte_tmp,reps=reps))

#%%
te_5000 = {'L2N':np.zeros(10,dtype=float), 'L2F':np.zeros(10,dtype=float), 'Prog-NN':np.zeros(10,dtype=float), 'DF-CNN':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float),'EWC':np.zeros(10,dtype=float), 'Online EWC':np.zeros(10,dtype=float), 'SI':np.zeros(10,dtype=float)}

for count,name in enumerate(te_5000.keys()):
    for i in range(10):
        te_5000[name][i] = tes_5000[count][i][9-i]

df_5000 = pd.DataFrame.from_dict(te_5000)
df_5000 = pd.melt(df_5000,var_name='Algorithms', value_name='Transfer Efficieny')

#%%
fig = plt.figure(constrained_layout=True,figsize=(16,16))
gs = fig.add_gridspec(16, 16)
clr = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
c = sns.color_palette(clr, n_colors=len(clr))

fontsize=24
ticksize=20
legendsize=14

ax = fig.add_subplot(gs[:7,:7])
for i, fte in enumerate(ftes_5000):
    if i == 0:
        ax.plot(np.arange(1,11), fte, color=clr[i], marker='.', markersize=12, label=alg_name[i], linewidth=3)
        continue

    if i == 1:
        ax.plot(np.arange(1,11), fte, color=clr[i], marker='.', markersize=12, label=alg_name[i], linewidth=3)
        continue
    
    ax.plot(np.arange(1,11), fte, color=clr[i], marker='.', markersize=12, label=alg_name[i])
    
ax.set_xticks(np.arange(1,11))
ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3,1.4])
ax.set_ylim(0.9, 1.37)
ax.tick_params(labelsize=ticksize)
# ax[0].legend(algos, loc='upper left', fontsize=14)
# ax[0].legend(algos, bbox_to_anchor=(1.2, -.2), loc=2, borderaxespad=0)

ax.set_ylabel('Forward Transfer Efficiency', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)


ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)

#ax[0][0].grid(axis='x')

ax = fig.add_subplot(gs[:7,8:15])
for i in range(task_num - 1):

    et = np.zeros((total_alg,task_num-i))

    for j in range(0,total_alg):
        et[j,:] = np.asarray(btes_5000[j][i])

    ns = np.arange(i + 1, task_num + 1)
    for j in range(0,total_alg):
        if j == 0:
            if i == 0:
                ax.plot(ns, et[j,:], marker='.', markersize=8, label = alg_name[j], color=clr[j], linewidth = 3)
            else:
                ax.plot(ns, et[j,:], marker='.', markersize=8, color=clr[j], linewidth = 3)
        elif j == 1:
            if i == 0:
                ax.plot(ns, et[j,:], marker='.', markersize=8, label = alg_name[j], color=clr[j], linewidth = 3)
            else:
                ax.plot(ns, et[j,:], marker='.', markersize=8, color=clr[j], linewidth = 3)
        else:
            if i == 0:
                ax.plot(ns, et[j,:], marker='.', markersize=8, label = alg_name[j], color=clr[j])
            else:
                ax.plot(ns, et[j,:], marker='.', markersize=8, color=clr[j])


# ax[1].set_title(ttle, fontsize=20)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('Backward Transfer Efficiency', fontsize=fontsize)
# ax.set_ylim(0.05 - 0.01, 0.5 + 0.01)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax[1].legend(loc='upper left', fontsize=12)
#ax[0][1].legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=22)
ax.set_yticks([.4,.6,.8,.9,1, 1.1,1.2])
ax.set_xticks(np.arange(1,11))
ax.set_ylim(0.85, 1.19)
ax.tick_params(labelsize=ticksize)
#ax[0][1].grid(axis='x')

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5)

##############################
ax = fig.add_subplot(gs[8:,:7])
ax.tick_params(labelsize=22)
#ax_ = sns.stripplot(x="Algorithms", y="Transfer Efficieny", data=df, palette=c, size=6, ax=ax[1][1])
ax.hlines(1, -1,8, colors='grey', linestyles='dashed',linewidth=1.5)
#sns.boxplot(x="Algorithms", y="Transfer Efficieny", data=mean_df, palette=c, linewidth=3, ax=ax[1][1])
#ax_=sns.pointplot(x="Algorithms", y="Transfer Efficieny", data=df_5000, join=False, color='grey', linewidth=1.5, ci='sd',ax=ax)
#ax_.set_yticks([.4,.6,.8,1, 1.2,1.4])
ax_ = sns.boxplot(
    x="Algorithms", y="Transfer Efficieny", data=df_5000, palette=c, 
    whis=np.inf, ax=ax,  showfliers=False, notch=1
    )
ax_.set_xlabel('', fontsize=fontsize)
ax.set_ylabel('Final Transfer Efficiency', fontsize=fontsize)
ax_.set_xticklabels(
    ['L2N','L2F','Prog-NN','DF-CNN','LwF','EWC','O-EWC','SI'],
    fontsize=20,rotation=45,ha="right",rotation_mode='anchor'
    )

stratified_scatter(te_5000,ax,16,c)


right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

######################################
ax = fig.add_subplot(gs[8:,8:25])
mean_error, std_error = unpickle('../recruitment_exp/result/recruitment_exp_5000.pickle')
ns = 10*np.array([10, 50, 100, 200, 350, 500])
clr = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
colors = sns.color_palette(clr, n_colors=len(clr))

#labels = ['recruiting', 'Uncertainty Forest', 'hybrid', '50 Random', 'BF', 'building']
labels = ['L2F (building)', 'UF (new)', 'recruiting', 'hybrid']
algo = ['building', 'UF', 'recruiting', 'hybrid']
adjust = 0
for i,key in enumerate(algo):
    err = np.array(mean_error[key])
    ax.plot(ns, err, c=colors[i], label=labels[i])
    #ax.fill_between(ns, 
    #        acc + 1.96*np.array(std_error[key]), 
    #        acc - 1.96*np.array(std_error[key]), 
    #        where=acc + 1.96*np.array(std_error[key]) >= acc - 1.96*np.array(std_error[key]), 
    #        facecolor=colors[i], 
    #        alpha=0.15,
    #        interpolate=False)


#ax.set_title('CIFAR Recruitment Experiment', fontsize=30)
ax.set_xscale('log')
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_ylabel('Generalization Error (Task 10)', fontsize=fontsize)
ax.set_xlabel('')
ax.tick_params(labelsize=ticksize)
#ax.set_ylim(0.325, 0.575)
#ax.set_title("CIFAR Recruitment",fontsize=titlesize)
ax.set_xticks([])
ax.set_yticks([0.45, 0.55, 0.65,0.75])
#ax.set_ylim([0.43,0.62])
#ax.text(50, 1, "50", fontsize=ticksize)
ax.text(100, 0.420, "100", fontsize=ticksize)
ax.text(500, 0.420, "500", fontsize=ticksize)
ax.text(5000, 0.420, "5000", fontsize=ticksize)
ax.text(120, 0.404, "Number of Task 10 Samples", fontsize=fontsize)

ax.legend(loc='lower left',fontsize=legendsize, frameon=False)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)


plt.savefig('result/figs/benchmark_5000.pdf', dpi=500)
# %%