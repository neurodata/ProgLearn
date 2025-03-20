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
            print(err[j][i],j,i)
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
    multitask_df = unpickle(filename)
    err = []
    for ii in range(10):
        tmp = 1 - np.array(multitask_df[multitask_df['task']==ii+1]['task_1_accuracy'])
        err.append(tmp)

    return err

#%%
alg_name = ['SiLLy-N', 'Model Zoo', 'LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay', 'Partial Replay', 'None']
model_file = ['dnn0', 'model_zoo', 'LwF', 'EWC', 'OEWC', 'SI', 'er', 'agem', 'tag', 'offline', 'exact', 'None']
total_alg = 12
slots = 10
shifts = 6

#%% claculate TE for label shuffle
reps = slots*shifts
tes_label_shuffle = [[] for i in range(total_alg)]

for alg in range(total_alg): 
    err_ = np.zeros(10,dtype=float)

    for slot in range(slots):
        for shift in range(shifts):
            if alg < 1:
                filename = '/Users/jayantadey/progressive-learning/experiments/plot_label_shuffled_angle_recruitment/label_shuffle_result/'+model_file[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            elif alg == 1:
                filename = '/Users/jayantadey/progressive-learning/experiments/plot_label_shuffled_angle_recruitment/label_shuffle_result/'+model_file[alg]+'_'+str(slot+1)+'_'+str(shift+1)+'.pickle'
            else:
                filename = '/Users/jayantadey/progressive-learning/experiments/plot_label_shuffled_angle_recruitment/label_shuffle_result/'+model_file[alg]+'-'+str(slot+1)+'-'+str(shift+1)+'.pickle'

            err_ += np.ravel(np.array(get_error_matrix(filename)))
    
    err_ /= reps
    te = err_[0] / err_

    if alg == 2:
        tes_label_shuffle[alg].extend([1]*10)
    else:
        tes_label_shuffle[alg].extend(te)

#%% calculate TE for rotation experiment
alg_name = ['SiLLy-N', 'Model Zoo', 'LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay', 'Partial Replay', 'None']
model_file = ['dnn', 'model_zoo', 'LwF', 'EWC', 'OEWC', 'si', 'er', 'agem', 'tag', 'offline', 'exact', 'None']
total_alg = 12
angles = range(0,182,4)
tes_angle = [[] for i in range(total_alg)]

for alg in range(total_alg): 
    for angle in angles:
        if alg < 1:
            filename = '/Users/jayantadey/TPAMI_rebuttal/plots/ProgLearn/benchmarks/rotation_cifar/results/'+model_file[alg]+'-'+str(angle)+'.pickle'
        else:
            filename = '/Users/jayantadey/TPAMI_rebuttal/plots/ProgLearn/benchmarks/rotation_cifar/benchmarking_algorthms_result/'+model_file[alg]+'-'+str(angle)+'.pickle'

        err = unpickle(filename)
        tes_angle[alg].extend([err[0]/err[1]])


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

def calc_acc_per_task(err, total_task, reps):
#Tom Vient et al
    acc = []
    for ii in range(total_task):
        acc.append(1-err[total_task-1][ii]/reps)
    return acc

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
            # print(j,i)
            # print(err[j][i])
            bte[i].append((err[i][i]+1e-2)/(err[j][i]+1e-2))
            te[i].append((single_err[i]+1e-2)/(err[j][i]+1e-2))
                
    for i in range(total_task):
        fte.append((single_err[i]+1e-2)/(err[i][i]+1e-2))
            
            
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
    tasks = np.array(multitask_df['base_task'])
    task_list = [tasks[0]]

    jj = 0
    for ii in range(total_task-1):
        while 1:
            if tasks[jj] != task_list[-1]:
                break 
            jj += 1
        task_list.append(tasks[jj])


    err = [[] for _ in range(total_task)]

    for ii, id in enumerate(task_list):
        err[ii].extend(
            1 - np.array(
                multitask_df[multitask_df['base_task']==id]['accuracy']
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

def calc_final_acc(err, reps):
    avg_acc = []
    for err_ in err[-1]:
        avg_acc.append(1-err_/reps)
    
    return avg_acc
# %%
### MAIN HYPERPARAMS ###
slots = 10
task_num = 10
shifts = 6
total_alg = 6
combined_alg_name = ['unshuffled', 'shuffled 1', 'shuffled 2', 'shuffled 3', 'shuffled 4', 'shuffled 5']
btes = [[] for i in range(total_alg)]
ftes = [[] for i in range(total_alg)]
tes = [[] for i in range(total_alg)]
# model_file_combined = ['odin', 'model_zoo', 'LwF', 'offline', 'exact', 'None']
########################

#%% code for 500 samples
reps = slots*shifts
final_acc = []

for alg in range(total_alg): 
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for slot in range(slots):
        for shift in range(shifts):
            filename = '/Users/jayantadey/TPAMI_rebuttal/plots/ProgLearn/benchmarks/random_order_result_forest/dnn_'+str(shift)+'_'+str(slot)+'_'+str(alg)+'.pickle'
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
    fte, bte, te = get_fte_bte(err,single_err,task_num)
    avg_acc, avg_var = calc_avg_acc(err, task_num, reps)
    avg_single_acc, avg_single_var = calc_avg_single_acc(single_err, task_num, reps)
    final_acc.append(calc_acc_per_task(err, task_num, reps))

    btes[alg].extend(bte)
    ftes[alg].extend(fte)
    tes[alg].extend(te)
#%%
acc = {'unshuffled':np.zeros(10,dtype=float), 
        'shuffled 1':np.zeros(10,dtype=float), 'shuffled 2':np.zeros(10,dtype=float), 
        'shuffled 3':np.zeros(10,dtype=float), 'shuffled 4':np.zeros(10,dtype=float),
        'shuffled 5':np.zeros(10,dtype=float)}


for count,name in enumerate(acc.keys()):
    acc[name] = np.array(final_acc[count][::-1])

#%%
te = {'unshuffled':np.zeros(10,dtype=float), 
        'shuffled 1':np.zeros(10,dtype=float), 'shuffled 2':np.zeros(10,dtype=float), 
        'shuffled 3':np.zeros(10,dtype=float), 'shuffled 4':np.zeros(10,dtype=float),
        'shuffled 5':np.zeros(10,dtype=float)}

task_order = []
t = 1
for count,name in enumerate(te.keys()):
    for i in range(10):
        te[name][9-i] = np.log(tes[count][i][9-i])
        task_order.append(t)
        t += 1

mean_val = []
for name in te.keys():
    mean_val.append(np.mean(te[name]))
    print(name, np.round(np.mean(te[name]),2), np.round(np.std(te[name], ddof=1),2))

# arg = np.argsort(mean_val)[::-1]
# ordr.append(arg)
algos = list(te.keys())
# combined_alg_name = []

# for ii in arg:
#     combined_alg_name.append(
#         algos[ii]
#     )
    
# tmp_te = {}
# for id in combined_alg_name:
#     tmp_te[id] = te[id]

df_le = pd.DataFrame.from_dict(te)
df_le = pd.melt(df_le,var_name='Algorithms', value_name='Transfer Efficieny')
df_le.insert(2, "Task ID", task_order)


#####
tmp_acc = {}
for id in combined_alg_name:
    tmp_acc[id] = acc[id]

df_acc = pd.DataFrame.from_dict(tmp_acc)
df_acc = pd.melt(df_acc,var_name='Algorithms', value_name='Accuracy')

df_acc.insert(2, "Task ID", task_order)
#%% register the palettes from cifar
# clr = ['#e41a1c', '#4daf4a', '#984ea3', '#e41a1c', '#83d0c9', '#f781bf', '#b15928', '#f781bf', '#f47835', '#b15928', '#8b8589', '#4c516d']
# c_ = []
# universal_clr_dic = {}
# for id in ordr[0]:
#     c_.append(clr[id])

# for ii, name in enumerate(labels[0]):
#     print(name)
#     register_palette(name, clr[ii])
#     universal_clr_dic[name] = clr[ii]
# %%
alg_name = ['SiLLy-N','Model Zoo', 'LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay', 'Partial Replay', 'None']

fontsize=30
ticksize=22
# clr_ = sns.color_palette(
#                 '#e41a1c', 
#                 n_colors=10
#                 )

fig = plt.figure(constrained_layout=True,figsize=(25,6))
gs = fig.add_gridspec(6, 25)

clr = ["#e41a1c", "#e41a1c", "#e41a1c", "#e41a1c", "#e41a1c", "#e41a1c", "#e41a1c", "#e41a1c", "#e41a1c", "#e41a1c", "#e41a1c", "#e41a1c", "#e41a1c", "#e41a1c"]
c = sns.color_palette(clr, n_colors=10)

ax = fig.add_subplot(gs[:6,:6])
ax_ = sns.stripplot(x='Algorithms', y='Transfer Efficieny', data=df_le, hue='Task ID', palette=c, ax=ax, size=18, legend=None)
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x='Algorithms',
            y='Transfer Efficieny',
            data=df_le,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax)

ax.set_title("A. Task Order Shuffle", fontsize = fontsize)
ax_.set_xlabel('')
ax_.set_ylabel('Transfer',fontsize=fontsize)
ax_.tick_params('y', labelsize=ticksize)
ax_.set_yticks([-0.03, .15,.3])

right_side = ax_.spines["right"]
right_side.set_visible(False)
top_side = ax_.spines["top"]
top_side.set_visible(False)

clr = ["#e41a1c", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928"]
c = sns.color_palette(clr, n_colors=len(clr))
marker_style = ['.', 'v', '.', '.', '.', '+', 'o', '*', '.', '+', 'x', 'v']

ax = fig.add_subplot(gs[:6,7:13])

for alg_no,alg in enumerate(alg_name):
    if alg_no<1:
        ax.plot(np.arange(1,11),tes_label_shuffle[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3, marker=marker_style[alg_no])
    else:
        ax.plot(np.arange(1,11),tes_label_shuffle[alg_no], c=c[alg_no], label=alg_name[alg_no], marker=marker_style[alg_no])

ax.set_yticks([.8,.9,1,1.1,1.2])
ax.set_ylim([0.79,1.21])
ax.set_xticks(np.arange(1,11))

log_lbl = np.round(
    np.log([0.8,0.9, 1, 1.1, 1.2]),
    1
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)


ax.tick_params(labelsize=ticksize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)
ax.set_ylabel('Backward Transfer', fontsize=fontsize)
ax.set_title("B. Label Shuffled CIFAR", fontsize = fontsize)
ax.hlines(1,1,10, colors='grey', linestyles='dashed',linewidth=1.5)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
plt.tight_layout()


ax = fig.add_subplot(gs[:6,14:20])
angles = np.arange(0,184,4)
#alg_name = ['SynN','SynF','LwF','EWC','O-EWC','SI', 'Total Replay', 'Partial Replay', 'None']
#clr = ["#377eb8", "#e41a1c", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928"]
#c = sns.color_palette(clr, n_colors=len(clr))
#marker_style = ['.', '.', '.', '+', 'o', '*', '.', '+', 'o']

for alg_no,alg in enumerate(alg_name):
    if alg_no<2:
        ax.plot(angles,tes_angle[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3, marker=marker_style[alg_no])
    else:
        ax.plot(angles,tes_angle[alg_no], c=c[alg_no], label=alg_name[alg_no], marker=marker_style[alg_no])

ax.set_yticks([.6,.7,.8,.9,1,1.1])
ax.set_ylim([0.6,1.1])
ax.set_xticks([0,30,60,90,120,150,180])
ax.hlines(1,0,180, colors='grey', linestyles='dashed',linewidth=1.5)

log_lbl = np.round(
    np.log([.6,.7,.8,.9,1,1.1]),
    1
)
labels = [item.get_text() for item in ax.get_yticklabels()]

ax_.set_xticklabels(
    combined_alg_name,
    fontsize=fontsize,rotation=65,ha="right",rotation_mode='anchor'
    )

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)

ax.tick_params(labelsize=ticksize)
ax.set_xlabel('Angle of Rotation (Degrees)', fontsize=fontsize)
ax.set_ylabel('Backward Transfer', fontsize=fontsize)
ax.set_title("C. Rotation Experiment", fontsize=fontsize)
handles, labels_ = ax.get_legend_handles_labels()
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
plt.tight_layout()

fig.legend(handles, labels_, bbox_to_anchor=(.99, .93), fontsize=20, frameon=False)
plt.savefig('/Users/jayantadey/TPAMI_rebuttal/plots/ProgLearn/benchmarks/plot_adversary_taskOrder/adversary.pdf', dpi=500)


# %%
