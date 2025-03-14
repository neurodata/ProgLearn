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
def calc_acc_per_task(err, total_task, reps):
#Tom Vient et al
    acc = []
    for ii in range(total_task):
        acc.append(1-err[total_task-1][ii]/reps)
    return acc

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
            bte[i].append(err[i][i]/(err[j][i]+1e-20))
            te[i].append(single_err[i]/(err[j][i]+1e-20))
                
    for i in range(total_task):
        fte.append(single_err[i]/(err[i][i]+1e-20))
            
            
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
tes, ftes, btes, acc = [], [], [], []
reps = 20
count = 0
task_num = 110

for rep in range(reps):
    filename =  '/Users/jayantadey/TPAMI_rebuttal/ProgLearn/benchmarks/core50/results/sillyN_'+str(rep)+'fixed_20.pickle'
    multitask_df, single_task_df = unpickle(filename)
    single_err_, err_ = get_error_matrix(filename, task_num)

    if count == 0:
        single_err, err = single_err_, err_
    else:
        err = sum_error_matrix(err, err_, task_num)
        single_err = list(
            np.asarray(single_err) + np.asarray(single_err_)
        )

    count += 1

fte_, bte_, te_ = get_fte_bte(err,single_err, total_task=task_num)
avg_acc_, avg_var_ = calc_avg_acc(err, task_num, reps)
#avg_single_acc_, avg_single_var_ = calc_avg_single_acc(single_err, 50, reps)

tes.append(te_)
ftes.append(fte_)
btes.append(bte_)
acc.append(
    calc_acc_per_task(err,task_num,reps)
)
# %%
res_df = {'Transfer':np.zeros(task_num,dtype=float),
         'Forward\n Transfer':np.zeros(task_num,dtype=float),
         'Backward\n Transfer':np.zeros(task_num,dtype=float),
         'Accuracy':np.zeros(task_num,dtype=float)}

task_order =[]
t=1
for i in range(task_num):
    res_df['Transfer'][i] = np.log(tes[0][i][task_num-1-i])
    res_df['Backward\n Transfer'][i] = np.log(btes[0][i][task_num-1-i])
    res_df['Forward\n Transfer'][i] = np.log(ftes[0][i])
    task_order.append(t)
    t += 1

t=1
for i in range(task_num):
    res_df['Backward\n Transfer'][i] = np.log(btes[0][i][task_num-1-i])
    task_order.append(t)
    t += 1

t=1
for i in range(task_num):
    res_df['Forward\n Transfer'][i] = np.log(ftes[0][i])
    task_order.append(t)
    t += 1

t=1
for i in range(task_num):
    res_df['Accuracy'][i] = acc[0][i]
    task_order.append(t)
    t += 1
# %%
df = pd.DataFrame.from_dict(res_df)
df = pd.melt(df,var_name='Statistics', value_name='Value')
df.insert(2, "Task ID", task_order)
# %%
universal_clr_dict = {'reds': 'r'}

for ii, name in enumerate(universal_clr_dict.keys()):
    print(name)
    register_palette(name, universal_clr_dict[name])
#%%
clr =  sns.color_palette(
            'reds', 
             n_colors=task_num
            )

#%%
ticksize = 40
labelsize = 40
fontsize = 32
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

ax_ = sns.stripplot(x='Statistics', y='Value', data=df, hue='Task ID', palette=clr, ax=ax, size=25, legend=None)
ax_.set_xticklabels(
    df.keys(),
    fontsize=labelsize,rotation=0,rotation_mode='anchor'
    )
ax_.set_ylabel('Statistics Value', fontsize=labelsize)
ax_.set_xlabel('', fontsize=labelsize)
ax_.set_yticks([0,1,6])
ax_.tick_params(labelsize=ticksize)
ax_.set_xticklabels(
    res_df.keys(),
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )
#ax_.set_title('food1k', fontsize=labelsize+10)

right_side = ax_.spines["right"]
right_side.set_visible(False)
top_side = ax_.spines["top"]
top_side.set_visible(False)

plt.savefig('core50.pdf', bbox_inches='tight')
# %%