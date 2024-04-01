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

def calc_forget(err, total_task, reps):
#Tom Vient et al
    forget = 0
    for ii in range(total_task-1):
        forget += err[ii][ii] - err[total_task-1][ii]

    forget /= (total_task-1)
    return forget/reps

def calc_transfer(err, single_err, total_task, reps):
#Tom Vient et al
    transfer = np.zeros(total_task,dtype=float)

    for ii in range(total_task):
        transfer[ii] = (single_err[ii] - err[total_task-1][ii])/reps

    return np.mean(transfer)

def calc_acc(err, total_task, reps):
#Tom Vient et al
    acc = 0
    for ii in range(total_task):
        acc += (1-err[total_task-1][ii]/reps)
    return acc/total_task

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def calc_avg_acc(err, total_task, reps):
    avg_acc = np.zeros(total_task, dtype=float)
    avg_var = np.zeros(total_task, dtype=float)
    for i in range(total_task):
        avg_acc[i] = (1*(i+1) - np.sum(err[i])/reps + (4-i)*.1)/10
        avg_var[i] = np.var(1-np.array(err[i])/reps)
    return avg_acc, avg_var

def calc_avg_single_acc(err, total_task, reps):
    avg_acc = np.zeros(total_task, dtype=float)
    avg_var = np.zeros(total_task, dtype=float)
    for i in range(total_task):
        avg_acc[i] = (1*(i+1) - np.sum(err[:i+1])/reps + (total_task-1-i)*.1)/10
        avg_var[i] = np.var(1-np.array(err[:i+1])/reps)
    return avg_acc, avg_var
    
def get_fte_bte(err, single_err, total_task):
    bte = [[] for i in range(total_task)]
    te = [[] for i in range(total_task)]
    fte = []
    
    for i in range(total_task):
        for j in range(i,total_task):
            #print(err[j][i],j,i)
            bte[i].append(err[i][i]/err[j][i])
            te[i].append(single_err[i]/err[j][i])
                
    for i in range(total_task):
        fte.append(single_err[i]/err[i][i])
            
            
    return fte,bte,te

def calc_mean_bte(btes,task_num,reps=10):
    mean_bte = [[] for i in range(task_num)]


    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(btes[i][j])
        
        tmp=tmp/reps
        mean_bte[j].extend(tmp)
            
    return mean_bte     

def calc_mean_te(tes,task_num,reps=10):
    mean_te = [[] for i in range(task_num)]

    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(tes[i][j])
        
        tmp=tmp/reps
        mean_te[j].extend(tmp)
                                             
    return mean_te 

def calc_mean_fte(ftes,task_num,reps=1):
    fte = np.asarray(ftes)
    
    return list(np.mean(np.asarray(fte),axis=0))

def get_error_matrix(filename, total_task):
    multitask_df, single_task_df = unpickle(filename)

    err = [[] for _ in range(total_task)]

    for ii in range(total_task):
        err[ii].extend(
            1 - np.array(
                multitask_df[multitask_df['base_task']==ii+1]['accuracy']
                )
            )
    single_err = 1 - np.array(single_task_df['accuracy'])

    return single_err, err

def sum_error_matrix(error_mat1, error_mat2, total_task):
    err = [[] for _ in range(total_task)]

    for ii in range(total_task):
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

# %%
tes = []
budgets = [5,6,7,10,20,30,40,50]
reps = 1

for budget in budgets:
    filename =  '/Users/jayantadey/budgeted_proglearn/ProgLearn/benchmarks/food1k/results/synn_0_'+ str(budget) +'.pickle'
    multitask_df, single_task_df = unpickle(filename)
    single_err, err = get_error_matrix(filename, 50)

    fte_, bte_, te_ = get_fte_bte(err,single_err, total_task=50)
    avg_acc_, avg_var_ = calc_avg_acc(err, 50, reps)
    avg_single_acc_, avg_single_var_ = calc_avg_single_acc(single_err, 50, reps)

    tes.append(te_)
# %%
te_df = {'Budget 5':np.zeros(50,dtype=float), 'Budget 6':np.zeros(50,dtype=float),
         'Budget 7':np.zeros(50,dtype=float),
         'Budget 10':np.zeros(50,dtype=float), 'Budget 20':np.zeros(50,dtype=float), 
         'Budget 30':np.zeros(50,dtype=float), 'Budget 40':np.zeros(50,dtype=float),
         'Budget 50':np.zeros(50,dtype=float)}

task_order =[]
t=1
for count,name in enumerate(te_df.keys()):
    for i in range(50):
        te_df[name][49-i] = np.log(tes[count][i][49-i])
        task_order.append(t)
        t += 1

for name in te_df.keys():
    print(name, np.round(np.mean(te_df[name]),2), np.round(np.std(te_df[name], ddof=1),2))
# %%
df = pd.DataFrame.from_dict(te_df)
df = pd.melt(df,var_name='Algorithms', value_name='Transfer Efficieny')
df.insert(2, "Task ID", task_order)
# %%
universal_clr_dict = {'Budget 5': '#377eb8',
                      'Budget 6': '#377eb8',
                      'Budget 7': '#377eb8',
                      'Budget 10': '#377eb8',
                      'Budget 20': '#e41a1c',
                      'Budget 30': '#984ea3',
                      'Budget 40': '#f781bf',
                      'Budget 50': '#4daf4a'}

for ii, name in enumerate(universal_clr_dict.keys()):
    print(name)
    register_palette(name, universal_clr_dict[name])
#%%
c_combined = []
for name in te_df.keys():
    c_combined.extend(
        sns.color_palette(
            name, 
            n_colors=50
            )
        )

#%%
ticksize = 20
labelsize = 20
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

ax_ = sns.stripplot(x='Algorithms', y='Transfer Efficieny', data=df, hue='Task ID', palette=c_combined, ax=ax, size=25, legend=None)
ax_.set_xticklabels(
    te_df.keys(),
    fontsize=labelsize,rotation=65,ha="right",rotation_mode='anchor'
    )
ax.set_ylabel('Transfer', fontsize=labelsize)
ax.set_xlabel('', fontsize=labelsize)
ax_.set_yticks([0,.4])
ax.tick_params(axis='y', labelsize=ticksize)
ax.set_title('food1k', fontsize=30)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(0, 0, 4, colors='grey', linestyles='dashed',linewidth=1.5)

plt.savefig('food1k_budgeted.pdf')
# %%