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

#%%
btes_all = {}
ftes_all = {}
tes_all = {}
te_scatter = {}
df_fle_all = {}
df_ble_all = {}
df_le_all = {}
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
        filename = '/Users/jayantadey/ProgLearn/benchmarks/five_datasets/results/'+model_file_combined[alg]+'.pickle'

        multitask_df, single_task_df = unpickle(filename)

        single_err_, err_ = get_error_matrix(filename, task_num)

        if count == 0:
            single_err, err = single_err_, err_
        else:
            err = sum_error_matrix(err, err_, total_task=task_num)
            single_err = list(
                np.asarray(single_err) + np.asarray(single_err_)
            )

        count += 1
    #single_err /= reps
    #err /= reps
    fte, bte, te = get_fte_bte(err,single_err, total_task=task_num)
    avg_acc_, avg_var_ = calc_avg_acc(err, task_num, reps)
    avg_single_acc_, avg_single_var_ = calc_avg_single_acc(single_err, task_num, reps)

    btes[alg].extend(bte)
    ftes[alg].extend(fte)
    tes[alg].extend(te)
    avg_acc[alg]= avg_acc_
    avg_var[alg] = avg_var_
    avg_single_acc[alg]= avg_single_acc_
    avg_single_var[alg] = avg_single_var_

    print('Algo name:' , combined_alg_name[alg])
    print('Accuracy', np.round(calc_acc(err, task_num, reps),2))
    print('forget', np.round(calc_forget(err, task_num, reps),2))
    print('transfer', np.round(calc_transfer(err, single_err, task_num, reps),2))

#%%
fte_dict = {'SynN':np.zeros(5,dtype=float), 'SynF':np.zeros(5,dtype=float), 'model_zoo':np.zeros(5,dtype=float), 
    'LwF':np.zeros(5,dtype=float), 'EWC':np.zeros(5,dtype=float), 
    'O-EWC':np.zeros(5,dtype=float), 'SI':np.zeros(5,dtype=float),
    'ER':np.zeros(5,dtype=float), 'A-GEM':np.zeros(5,dtype=float),
    'TAG':np.zeros(5,dtype=float),
    'Total Replay':np.zeros(5,dtype=float), 'Partial Replay':np.zeros(5,dtype=float), 
    'None':np.zeros(5,dtype=float)}

task_order = []
for count,name in enumerate(fte_dict.keys()):
    for i in range(5):
        fte_dict[name][i] = np.log(ftes[count][i])
        task_order.append(i+1)

for name in fte_dict.keys():
    print(name, np.round(np.mean(fte_dict[name]),2), np.round(np.std(fte_dict[name], ddof=1),2))


df_fle = pd.DataFrame.from_dict(fte_dict)
df_fle = pd.melt(df_fle,var_name='Algorithms', value_name='Forward Transfer Efficieny')
df_fle.insert(2, "Task ID", task_order)
#%%
bte_dict = {'SynN':np.zeros(5,dtype=float), 'SynF':np.zeros(5,dtype=float), 'model_zoo':np.zeros(5,dtype=float), 
    'LwF':np.zeros(5,dtype=float), 'EWC':np.zeros(5,dtype=float), 
    'O-EWC':np.zeros(5,dtype=float), 'SI':np.zeros(5,dtype=float),
    'ER':np.zeros(5,dtype=float), 'A-GEM':np.zeros(5,dtype=float),
    'TAG':np.zeros(5,dtype=float),
    'Total Replay':np.zeros(5,dtype=float), 'Partial Replay':np.zeros(5,dtype=float), 
    'None':np.zeros(5,dtype=float)}

for count,name in enumerate(bte_dict.keys()):
    for i in range(5):
        bte_dict[name][i] = np.log(btes[count][i][4-i])

for name in bte_dict.keys():
    print(name, np.round(np.mean(bte_dict[name]),2), np.round(np.std(bte_dict[name], ddof=1),2))


df_ble = pd.DataFrame.from_dict(bte_dict)
df_ble = pd.melt(df_ble,var_name='Algorithms', value_name='Backward Transfer Efficieny')
df_ble.insert(2, "Task ID", task_order)
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

for name in te.keys():
    print(name, np.round(np.mean(te[name]),2), np.round(np.std(te[name], ddof=1),2))


df = pd.DataFrame.from_dict(te)
df = pd.melt(df,var_name='Algorithms', value_name='Transfer Efficieny')
df.insert(2, "Task ID", task_order)
#%%
btes_all['five_dataset'] = btes
ftes_all['five_dataset'] = ftes
df_fle_all['five_dataset'] = df_fle
df_ble_all['five_dataset'] = df_ble
df_le_all['five_dataset'] = df
tes_all['five_dataset'] = tes
te_scatter['five_dataset'] = te

#%%

### MAIN HYPERPARAMS ###
task_num = 20
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

#%% 
reps = 1

for alg in range(total_alg): 
    count = 0 

    for rep in range(reps):
        filename = '/Users/jayantadey/ProgLearn/benchmarks/mini_imagenet/results/'+model_file_combined[alg]+'.pickle'

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
    #single_err /= reps
    #err /= reps
    fte, bte, te = get_fte_bte(err,single_err, task_num)
    avg_acc_, avg_var_ = calc_avg_acc(err, task_num, reps)
    avg_single_acc_, avg_single_var_ = calc_avg_single_acc(single_err, task_num, reps)

    btes[alg].extend(bte)
    ftes[alg].extend(fte)
    tes[alg].extend(te)
    avg_acc[alg]= avg_acc_
    avg_var[alg] = avg_var_
    avg_single_acc[alg]= avg_single_acc_
    avg_single_var[alg] = avg_single_var_

    print('Algo name:' , combined_alg_name[alg])
    print('Accuracy', np.round(calc_acc(err, task_num, reps),2))
    print('forget', np.round(calc_forget(err, task_num, reps),2))
    print('transfer', np.round(calc_transfer(err, single_err, task_num, reps),2))

#%%
fte_dict = {'SynN':np.zeros(20,dtype=float), 'SynF':np.zeros(20,dtype=float), 'model_zoo':np.zeros(20,dtype=float), 
    'LwF':np.zeros(20,dtype=float), 'EWC':np.zeros(20,dtype=float), 
    'O-EWC':np.zeros(20,dtype=float), 'SI':np.zeros(20,dtype=float),
    'ER':np.zeros(20,dtype=float), 'A-GEM':np.zeros(20,dtype=float),
    'TAG':np.zeros(20,dtype=float),
    'Total Replay':np.zeros(20,dtype=float), 'Partial Replay':np.zeros(20,dtype=float), 
    'None':np.zeros(20,dtype=float)}

task_order = []
for count,name in enumerate(fte_dict.keys()):
    for i in range(20):
        fte_dict[name][i] = np.log(ftes[count][i])
        task_order.append(i+1)

for name in fte_dict.keys():
    print(name, np.round(np.mean(fte_dict[name]),2), np.round(np.std(fte_dict[name], ddof=1),2))


df_fle = pd.DataFrame.from_dict(fte_dict)
df_fle = pd.melt(df_fle,var_name='Algorithms', value_name='Forward Transfer Efficieny')
df_fle.insert(2, "Task ID", task_order)
#%%
bte_dict = {'SynN':np.zeros(20,dtype=float), 'SynF':np.zeros(20,dtype=float), 'model_zoo':np.zeros(20,dtype=float), 
    'LwF':np.zeros(20,dtype=float), 'EWC':np.zeros(20,dtype=float), 
    'O-EWC':np.zeros(20,dtype=float), 'SI':np.zeros(20,dtype=float),
    'ER':np.zeros(20,dtype=float), 'A-GEM':np.zeros(20,dtype=float),
    'TAG':np.zeros(20,dtype=float),
    'Total Replay':np.zeros(20,dtype=float), 'Partial Replay':np.zeros(20,dtype=float), 
    'None':np.zeros(20,dtype=float)}

for count,name in enumerate(bte_dict.keys()):
    for i in range(20):
        bte_dict[name][i] = np.log(btes[count][i][19-i])

for name in bte_dict.keys():
    print(name, np.round(np.mean(bte_dict[name]),2), np.round(np.std(bte_dict[name], ddof=1),2))


df_ble = pd.DataFrame.from_dict(bte_dict)
df_ble = pd.melt(df_ble,var_name='Algorithms', value_name='Backward Transfer Efficieny')
df_ble.insert(2, "Task ID", task_order)
#%%
te = {'SynN':np.zeros(20,dtype=float), 'SynF':np.zeros(20,dtype=float), 'model_zoo':np.zeros(20,dtype=float), 
    'LwF':np.zeros(20,dtype=float), 'EWC':np.zeros(20,dtype=float), 
    'O-EWC':np.zeros(20,dtype=float), 'SI':np.zeros(20,dtype=float),
    'ER':np.zeros(20,dtype=float), 'A-GEM':np.zeros(20,dtype=float),
    'TAG':np.zeros(20,dtype=float),
    'Total Replay':np.zeros(20,dtype=float), 'Partial Replay':np.zeros(20,dtype=float), 
    'None':np.zeros(20,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(20):
        te[name][i] = np.log(tes[count][i][19-i])


for name in te.keys():
    print(name, np.round(np.mean(te[name]),2), np.round(np.std(te[name], ddof=1),2))

df = pd.DataFrame.from_dict(te)
df = pd.melt(df,var_name='Algorithms', value_name='Transfer Efficieny')
df.insert(2, "Task ID", task_order)

#%%
btes_all['imagenet'] = btes
ftes_all['imagenet'] = ftes
df_fle_all['imagenet'] = df_fle
df_ble_all['imagenet'] = df_ble
df_le_all['imagenet'] = df
tes_all['imagenet'] = tes
te_scatter['imagenet'] = te
#%%

### MAIN HYPERPARAMS ###
task_num = 50
total_alg = 4
combined_alg_name = ['SynN','SynF', 'Model Zoo', 'LwF']
btes = [[] for i in range(total_alg)]
ftes = [[] for i in range(total_alg)]
tes = [[] for i in range(total_alg)]
model_file_combined = ['synn','synf', 'model_zoo', 'LwF']
avg_acc = [[] for i in range(total_alg)]
avg_var = [[] for i in range(total_alg)]
avg_single_acc = [[] for i in range(total_alg)]
avg_single_var = [[] for i in range(total_alg)]
########################

#%% 

for alg in range(total_alg): 

    filename = '/Users/jayantadey/ProgLearn/benchmarks/food1k/results/'+model_file_combined[alg]+'.pickle'

    multitask_df, single_task_df = unpickle(filename)

    single_err, err = get_error_matrix(filename, task_num)

    #single_err /= reps
    #err /= reps
    fte, bte, te = get_fte_bte(err,single_err, task_num)
    avg_acc_, avg_var_ = calc_avg_acc(err, task_num, 1)
    avg_single_acc_, avg_single_var_ = calc_avg_single_acc(single_err, task_num, 1)

    btes[alg].extend(bte)
    ftes[alg].extend(fte)
    tes[alg].extend(te)
    avg_acc[alg]= avg_acc_
    avg_var[alg] = avg_var_
    avg_single_acc[alg]= avg_single_acc_
    avg_single_var[alg] = avg_single_var_

    print('Algo name:' , combined_alg_name[alg])
    print('Accuracy', np.round(calc_acc(err, task_num, 1),2))
    print('forget', np.round(calc_forget(err, task_num, 1),2))
    print('transfer', np.round(calc_transfer(err, single_err, task_num, 1),2))

#%%
fte_dict = {'SynN':np.zeros(50,dtype=float), 'SynF':np.zeros(50,dtype=float), 'model_zoo':np.zeros(50,dtype=float), 
    'LwF':np.zeros(50,dtype=float)}

task_order = []
for count,name in enumerate(fte_dict.keys()):
    for i in range(50):
        fte_dict[name][i] = np.log(ftes[count][i])
        task_order.append(i+1)

for name in fte_dict.keys():
    print(name, np.round(np.mean(fte_dict[name]),2), np.round(np.std(fte_dict[name], ddof=1),2))


df_fle = pd.DataFrame.from_dict(fte_dict)
df_fle = pd.melt(df_fle,var_name='Algorithms', value_name='Forward Transfer Efficieny')
df_fle.insert(2, "Task ID", task_order)
#%%
bte_dict = {'SynN':np.zeros(50,dtype=float), 'SynF':np.zeros(50,dtype=float), 'model_zoo':np.zeros(50,dtype=float), 
    'LwF':np.zeros(50,dtype=float)}

for count,name in enumerate(bte_dict.keys()):
    for i in range(50):
        bte_dict[name][i] = np.log(btes[count][i][49-i])

for name in bte_dict.keys():
    print(name, np.round(np.mean(bte_dict[name]),2), np.round(np.std(bte_dict[name], ddof=1),2))


df_ble = pd.DataFrame.from_dict(bte_dict)
df_ble = pd.melt(df_ble,var_name='Algorithms', value_name='Backward Transfer Efficieny')
df_ble.insert(2, "Task ID", task_order)
#%%
te = {'SynN':np.zeros(50,dtype=float), 'SynF':np.zeros(50,dtype=float), 'model_zoo':np.zeros(50,dtype=float), 
    'LwF':np.zeros(50,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(50):
        te[name][i] = np.log(tes[count][i][49-i])


for name in te.keys():
    print(name, np.round(np.mean(te[name]),2), np.round(np.std(te[name], ddof=1),2))

df = pd.DataFrame.from_dict(te)
df = pd.melt(df,var_name='Algorithms', value_name='Transfer Efficieny')
df.insert(2, "Task ID", task_order)


#%%
btes_all['food1k'] = btes
ftes_all['food1k'] = ftes
df_fle_all['food1k'] = df_fle
df_ble_all['food1k'] = df_ble
df_le_all['food1k'] = df
tes_all['food1k'] = tes
te_scatter['food1k'] = te

# %%
fig = plt.figure(constrained_layout=True,figsize=(34,28))
gs = fig.add_gridspec(28,34)


tes = tes_all['five_dataset']
btes = btes_all['five_dataset']
ftes = ftes_all['five_dataset']
df = df_le_all['five_dataset']
te = te_scatter['five_dataset']

total_alg=13
task_num=5
combined_alg_name = ['SynN','SynF', 'Model Zoo', 'LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay', 'Partial Replay', 'None']
marker_style = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']
marker_style_scatter = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']

c_combined = sns.color_palette('Reds', n_colors=task_num)

fontsize=30
ticksize=26
legendsize=20

ax = fig.add_subplot(gs[4:12,:8])

ax_ = sns.stripplot(x='Algorithms', y='Forward Transfer Efficieny', data=df_fle_all['five_dataset'], hue='Task ID', palette=c_combined, ax=ax, size=25, legend=None)
ax_.set_xticklabels(
    combined_alg_name,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )

ax.set_xlabel('')
ax.set_ylabel('Forward Transfer', fontsize=fontsize)
ax_.set_yticks([.3, 0,-2])
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(0, 0,12, colors='grey', linestyles='dashed',linewidth=1.5)

ax.set_title('5-dataset', fontsize=fontsize+20)

handles, labels_ = ax.get_legend_handles_labels()

###########################################################
ax = fig.add_subplot(gs[15:23,:8])

ax_ = sns.stripplot(x='Algorithms', y='Backward Transfer Efficieny', data=df_ble_all['five_dataset'], hue='Task ID', palette=c_combined, ax=ax, size=25, legend=None)
ax_.set_xticklabels(
    combined_alg_name,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )

ax.set_xlabel('')
ax.set_ylabel('Backward Transfer', fontsize=fontsize)
ax_.set_yticks([1, 0,-3])
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(0, 0, 12, colors='grey', linestyles='dashed',linewidth=1.5)


tes = tes_all['imagenet']
btes = btes_all['imagenet']
ftes = ftes_all['imagenet']
df = df_le_all['imagenet']
te = te_scatter['imagenet']

task_num = 20
total_alg = 13
combined_alg_name = ['SynN','SynF', 'Model Zoo', 'LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay', 'Partial Replay', 'None']
marker_style = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']
marker_style_scatter = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']

c_combined = sns.color_palette('Reds', n_colors=total_alg)

ax = fig.add_subplot(gs[4:12,12:20])

ax_ = sns.stripplot(x='Algorithms', y='Forward Transfer Efficieny', data=df_fle_all['imagenet'], hue='Task ID', palette=c_combined, ax=ax, size=25, legend=None)
ax_.set_xticklabels(
    combined_alg_name,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )

ax.set_xlabel('')
ax.set_ylabel('Forward Transfer', fontsize=fontsize)
ax.set_title('Split Mini-Imagenet\n (2400 samples)', fontsize=fontsize+20)
ax_.set_yticks([.6, 0,-.4])
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(0, 0,12, colors='grey', linestyles='dashed',linewidth=1.5)


ax = fig.add_subplot(gs[15:23,12:20])

ax_ = sns.stripplot(x='Algorithms', y='Backward Transfer Efficieny', data=df_ble_all['imagenet'], hue='Task ID', palette=c_combined, ax=ax, size=25, legend=None)
ax_.set_xticklabels(
    combined_alg_name,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )

ax.set_xlabel('')
ax.set_ylabel('Backward Transfer', fontsize=fontsize)
ax_.set_yticks([.2, 0,-.6])
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(0, 0, 12, colors='grey', linestyles='dashed',linewidth=1.5)

###############################################
fig.legend(handles, labels_, bbox_to_anchor=(.97, .7), fontsize=legendsize+12, frameon=False)

tes = tes_all['food1k']
btes = btes_all['food1k']
ftes = ftes_all['food1k']
df = df_le_all['food1k']
te = te_scatter['food1k']

task_num = 50
total_alg = 4
combined_alg_name = ['SynN','SynF', 'Model Zoo', 'LwF']

marker_style = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']
marker_style_scatter = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']
c_combined = sns.color_palette('Reds', n_colors=total_alg)

ax = fig.add_subplot(gs[4:12,24:32])

ax_ = sns.stripplot(x='Algorithms', y='Forward Transfer Efficieny', data=df_fle_all['food1k'], hue='Task ID', palette=c_combined, ax=ax, size=25, legend=None)
ax_.set_xticklabels(
    combined_alg_name,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )

ax.set_xlabel('')
ax.set_ylabel('Forward Transfer', fontsize=fontsize)
ax.set_title('FOOD1k 50X20\n (1200 samples)', fontsize=fontsize+20)
ax_.set_yticks([.4, 0,-.1])
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(0, 0, 4, colors='grey', linestyles='dashed',linewidth=1.5)

ax = fig.add_subplot(gs[15:23,24:32])

ax_ = sns.stripplot(x='Algorithms', y='Backward Transfer Efficieny', data=df_ble_all['food1k'], hue='Task ID', palette=c_combined, ax=ax, size=25, legend=None)
ax_.set_xticklabels(
    combined_alg_name,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )

ax.set_xlabel('')
ax.set_ylabel('Backward Transfer', fontsize=fontsize)
ax_.set_yticks([.3, 0,-.3])
ax.tick_params(labelsize=ticksize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(0, 0, 4, colors='grey', linestyles='dashed',linewidth=1.5)

plt.savefig('combined.pdf')
# %%