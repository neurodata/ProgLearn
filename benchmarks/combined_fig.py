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
df_all = {}
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
        filename = 'five_datasets/results/'+model_file_combined[alg]+'.pickle'

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

#%%
btes_all['five_dataset'] = btes
ftes_all['five_dataset'] = ftes
df_all['five_dataset'] = df
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
        filename = 'mini_imagenet/results/'+model_file_combined[alg]+'.pickle'

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


#%%
btes_all['imagenet'] = btes
ftes_all['imagenet'] = ftes
df_all['imagenet'] = df
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

    filename = 'food1k/results/'+model_file_combined[alg]+'.pickle'

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
te = {'SynN':np.zeros(50,dtype=float), 'SynF':np.zeros(50,dtype=float), 'model_zoo':np.zeros(50,dtype=float), 
    'LwF':np.zeros(50,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(50):
        te[name][i] = np.log(tes[count][i][49-i])


for name in te.keys():
    print(name, np.round(np.mean(te[name]),2), np.round(np.std(te[name], ddof=1),2))

df = pd.DataFrame.from_dict(te)
df = pd.melt(df,var_name='Algorithms', value_name='Transfer Efficieny')



#%%
btes_all['food1k'] = btes
ftes_all['food1k'] = ftes
df_all['food1k'] = df
tes_all['food1k'] = tes
te_scatter['food1k'] = te







# %%
fig = plt.figure(constrained_layout=True,figsize=(42,45))
gs = fig.add_gridspec(45,42)


tes = tes_all['five_dataset']
btes = btes_all['five_dataset']
ftes = ftes_all['five_dataset']
df = df_all['five_dataset']
te = te_scatter['five_dataset']

total_alg=13
task_num=5
combined_alg_name = ['SynN','SynF', 'Model Zoo', 'LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay', 'Partial Replay', 'None']
marker_style = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']
marker_style_scatter = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']

clr_combined = ["#377eb8", "#e41a1c", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928"]
c_combined = sns.color_palette(clr_combined, n_colors=total_alg)

fontsize=35
ticksize=26
legendsize=20

ax = fig.add_subplot(gs[4:12,:8])

for i, fte in enumerate(ftes):
    fte[0] = 1
    if i == 0:
        ax.plot(np.arange(1,6), fte, color=c_combined[i], marker=marker_style[i], markersize=12, label=combined_alg_name[i], linewidth=3)
        continue

    if i == 1:
        ax.plot(np.arange(1,6), fte, color=c_combined[i], marker=marker_style[i], markersize=12, label=combined_alg_name[i], linewidth=3)
        continue
    
    ax.plot(np.arange(1,6), fte, color=c_combined[i], marker=marker_style[i], markersize=12, label=combined_alg_name[i])

ax.set_title('Forward Learning (FL)', fontsize=fontsize+5)
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

ax.set_ylabel('log FLE', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,5, colors='grey', linestyles='dashed',linewidth=1.5)

handles, labels_ = ax.get_legend_handles_labels()

###########################################################
#ax[0][0].grid(axis='x')
ax = fig.add_subplot(gs[2:15,8:27], projection='3d')

#cmap = sns.color_palette("coolwarm", as_cmap=True)

for i in range(task_num - 1):

    et = np.zeros((total_alg,task_num-i))

    for j in range(0,total_alg):
        et[j,:] = np.asarray(btes[j][i])

    ns = np.arange(i + 1, task_num + 1)
    ns_new = np.linspace(ns.min(), ns.max(), 300)

    for j in range(0,total_alg):
        y_interp = np.interp(ns_new, ns, et[j,:])
        idx = np.where(y_interp < 1.0)[0]
        supper = y_interp.copy()
        supper[idx] = np.nan

        idx = np.where(y_interp >= 1.0)[0]
        slower = y_interp.copy()
        slower[idx] = np.nan

        ax.plot(ns_new, supper, zs=j, zdir='y', color='r', linewidth=3)
        ax.plot(ns_new, slower, zs=j, zdir='y', color='b', linewidth=3)
        

xs = np.linspace(0, 5, 100)
zs = np.linspace(0, 13, 100)
X, Y = np.meshgrid(xs, zs)
Z = np.ones(X.shape)

ax.plot_surface(X, Y, Z, color='grey', alpha=.3)

for ii in range(total_alg):
    zs = np.linspace(ii-.05,ii+.05,10)
    X, Y = np.meshgrid(xs, zs)
    Z = np.ones(X.shape)

    ax.plot_surface(X, Y, Z, color='grey', alpha=1)

ax.view_init(elev=10., azim=15, roll=0)

'''for i in range(total_alg_top,total_alg_top+total_alg_bottom-1):
    ax.plot(1,0,color=c_combined[i], marker=marker_style[i], markersize=8,label=combined_alg_name[i])'''

ax.text(.9, .5, 3.8, 'Backward Learning (BL)', fontsize=fontsize+5)
ax.text(.9, .75, 4.2, "5-dataset", fontsize=fontsize+20)

ax.set_xlabel('Tasks seen', fontsize=30, labelpad=15)
ax.set_zlabel('log BLE', fontsize=30, labelpad=15)

ax.set_zticks([.2,1,2,3])
ax.set_xticks(np.arange(1,6,1))
ax.set_yticks(np.arange(0,13,1))
ax.set_zlim(0.2, 3)
ax.set_ylim([0,12])
log_lbl = np.round(
    np.log([.2,1,2,3]),
    1
)
labels = [item.get_text() for item in ax.get_zticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_zticklabels(labels)
ax.set_yticklabels(combined_alg_name, rotation=80)
ax.tick_params(labelsize=ticksize-4)
#ax[0][1].grid(axis='x')
ax.invert_xaxis()


#ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

for ytick, color in zip(ax.get_yticklabels(), clr_combined):
    ytick.set_color(color)



#handles, labels_ = ax.get_legend_handles_labels()
#ax.legend(loc='center left', bbox_to_anchor=(.8, 0.5), fontsize=legendsize+16)


ax = fig.add_subplot(gs[4:12,28:36])
ax.tick_params(labelsize=22)
ax_ = sns.boxplot(
    x="Algorithms", y="Transfer Efficieny", data=df, palette=c_combined, whis=np.inf,
    ax=ax, showfliers=False, notch=1
    )
ax.hlines(0, -1,12, colors='grey', linestyles='dashed',linewidth=1.5)
#sns.boxplot(x="Algorithms", y="Transfer Efficieny", data=mean_df, palette=c, linewidth=3, ax=ax[1][1])
#ax_=sns.pointplot(x="Algorithms", y="Transfer Efficieny", data=df_500, join=False, color='grey', linewidth=1.5, ci='sd',ax=ax)
ax.set_yticks([-3, -1, 0, .5])
ax.set_title('Overall Learning', fontsize=fontsize+5)
ax_.set_xlabel('', fontsize=fontsize)
ax.set_ylabel('log LE after 5 Tasks', fontsize=fontsize-5)
ax_.set_xticklabels(
    ['SynN','SynF', 'Model zoo', 'LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay','Partial Replay', 'None'],
    fontsize=22,rotation=65,ha="right",rotation_mode='anchor'
    )

stratified_scatter(te,ax,16,c_combined,marker_style_scatter)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

fig.legend(handles, labels_, bbox_to_anchor=(.97, .7), fontsize=legendsize+12, frameon=False)

'''ax = fig.add_subplot(gs[16:24,8:16])

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
ax = fig.add_subplot(gs[16:24,20:28])

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
top_side.set_visible(False)'''

#fig.text(.45, 0.88, "Five Dataset", fontsize=fontsize+15)
#plt.savefig('five_dataset.pdf')



tes = tes_all['imagenet']
btes = btes_all['imagenet']
ftes = ftes_all['imagenet']
df = df_all['imagenet']
te = te_scatter['imagenet']

task_num = 20
total_alg = 13
combined_alg_name = ['SynN','SynF', 'Model Zoo', 'LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay', 'Partial Replay', 'None']
marker_style = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']
marker_style_scatter = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']

clr_combined = ["#377eb8", "#e41a1c", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928"]
c_combined = sns.color_palette(clr_combined, n_colors=total_alg)


ax = fig.add_subplot(gs[19:27,:8])

for i, fte in enumerate(ftes):
    fte[0] = 1
    if i == 0:
        ax.plot(np.arange(1,21), fte, color=c_combined[i], marker=marker_style[i], markersize=12, label=combined_alg_name[i], linewidth=3)
        continue

    if i == 1:
        ax.plot(np.arange(1,21), fte, color=c_combined[i], marker=marker_style[i], markersize=12, label=combined_alg_name[i], linewidth=3)
        continue
    
    ax.plot(np.arange(1,21), fte, color=c_combined[i], marker=marker_style[i], markersize=12, label=combined_alg_name[i])
    
ax.set_title('Forward Learning (FL)', fontsize=fontsize+5)
ax.set_xticks(np.arange(1,21,3))
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

ax.set_ylabel('log FLE', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,21, colors='grey', linestyles='dashed',linewidth=1.5)

handles, labels_ = ax.get_legend_handles_labels()

ax = fig.add_subplot(gs[17:30,8:27], projection='3d')

xs = np.linspace(0, task_num, 100)
for ii in range(total_alg):
    zs = np.linspace(ii-.05,ii+.05,10)
    X, Y = np.meshgrid(xs, zs)
    Z = np.ones(X.shape)

    ax.plot_surface(X, Y, Z, color='grey', alpha=1)

#cmap = sns.color_palette("coolwarm", as_cmap=True)
color = ['b', 'r']
for i in range(task_num - 1):

    et = np.zeros((total_alg,task_num-i))

    for j in range(0,total_alg):
        et[j,:] = np.asarray(btes[j][i])

    ns = np.arange(i + 1, task_num + 1)
    ns_new = np.linspace(ns.min(), ns.max(), 300)

    for j in range(0,total_alg):
        y_interp = np.interp(ns_new, ns, et[j,:])
        idx = np.where(y_interp < 1.0)[0]
        supper = y_interp.copy()
        supper[idx] = np.nan

        idx = np.where(y_interp >= 1.0)[0]
        slower = y_interp.copy()
        slower[idx] = np.nan

        ax.plot(ns_new, supper, zs=j, zdir='y', color='r', linewidth=3)
        ax.plot(ns_new, slower, zs=j, zdir='y', color='b', linewidth=3)
        

zs = np.linspace(0, total_alg-1, 100)
X, Y = np.meshgrid(xs, zs)
Z = np.ones(X.shape)

ax.plot_surface(X, Y, Z, color='grey', alpha=.3)

ax.view_init(elev=10., azim=15, roll=0)

'''for i in range(total_alg_top,total_alg_top+total_alg_bottom-1):
    ax.plot(1,0,color=c_combined[i], marker=marker_style[i], markersize=8,label=combined_alg_name[i])'''

ax.text(.9, .6, 1.75, 'Backward Learning (BL)', fontsize=fontsize+5)
ax.text(.1, .1, 2, "Split Mini-Imagenet(2400 samples)", fontsize=fontsize+20)
ax.set_xlabel('Tasks seen', fontsize=30, labelpad=15)
ax.set_zlabel('log BLE', fontsize=30, labelpad=15)

ax.set_zticks([.4,1,1.5])
ax.set_xticks(np.arange(4,task_num+1,8))
ax.set_yticks(np.arange(0,total_alg,1))
ax.set_zlim(0.4, 1.5)
ax.set_ylim([0,12])
log_lbl = np.round(
    np.log([.4,1,1.5]),
    1
)
labels = [item.get_text() for item in ax.get_zticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_zticklabels(labels)
ax.set_yticklabels(combined_alg_name, rotation=80)
ax.tick_params(labelsize=ticksize-4)
#ax[0][1].grid(axis='x')
ax.invert_xaxis()


#ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

for ytick, color in zip(ax.get_yticklabels(), clr_combined):
    ytick.set_color(color)

###############################################

ax = fig.add_subplot(gs[19:27,28:36])
ax.tick_params(labelsize=22)
ax_ = sns.boxplot(
    x="Algorithms", y="Transfer Efficieny", data=df, palette=c_combined, whis=np.inf,
    ax=ax, showfliers=False, notch=1
    )
ax.hlines(0, -1,12, colors='grey', linestyles='dashed',linewidth=1.5)
#sns.boxplot(x="Algorithms", y="Transfer Efficieny", data=mean_df, palette=c, linewidth=3, ax=ax[1][1])
#ax_=sns.pointplot(x="Algorithms", y="Transfer Efficieny", data=df_500, join=False, color='grey', linewidth=1.5, ci='sd',ax=ax)
#ax_.set_yticks([.4,.6,.8,1, 1.2,1.4])
ax.set_title('Overall Learning', fontsize=fontsize+5)
ax_.set_xlabel('', fontsize=fontsize)
ax.set_ylabel('log LE after 20 Tasks', fontsize=fontsize-5)
ax_.set_xticklabels(
    ['SynN','SynF', 'Model zoo', 'LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay','Partial Replay', 'None'],
    fontsize=22,rotation=65,ha="right",rotation_mode='anchor'
    )

stratified_scatter(te,ax,16,c_combined,marker_style_scatter)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

fig.legend(handles, labels_, bbox_to_anchor=(.97, .7), fontsize=legendsize+12, frameon=False)

'''ax = fig.add_subplot(gs[16:24,8:16])

for i in range(total_alg):
    if i==0 or i==1:
        ax.plot(np.arange(1,21,1) ,avg_acc[i], color=c_combined[i], marker=marker_style[i], linewidth=3)
    else:
        ax.plot(np.arange(1,21,1) ,avg_acc[i], color=c_combined[i], marker=marker_style[i])
    #ax.fill_between(np.arange(1,6,1), avg_acc[i]-1.96*avg_var[i], avg_acc[i]+1.96*avg_var[i], facecolor=c_combined[i], alpha=.3)

ax.hlines(.1, 1,21, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')
ax.set_yticks([.2,.4,.6,.8,1])
ax.set_xticks(np.arange(1,21,3))
ax.tick_params(labelsize=ticksize)
ax.set_ylabel('Accuracy[$\pm$ std dev.]', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

############################
ax = fig.add_subplot(gs[16:24,20:28])

for i in range(total_alg):
    if i==0 or i==1:
        ax.plot(np.arange(1,21,1) ,avg_single_acc[i], color=c_combined[i], marker=marker_style[i], linewidth=3)
    else:
        ax.plot(np.arange(1,21,1) ,avg_single_acc[i], color=c_combined[i], marker=marker_style[i])
   # ax.fill_between(np.arange(1,6,1), avg_single_acc[i]-1.96*avg_single_var[i], avg_single_acc[i]+1.96*avg_single_var[i], facecolor=c_combined[i], alpha=.3)

ax.hlines(.1, 1,21, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')
ax.set_yticks([.2,.4,.6,.8,1])
ax.set_xticks(np.arange(1,21,3))
ax.tick_params(labelsize=ticksize)
ax.set_ylabel('Single task accuracy', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)'''

#fig.text(.4, 0.88, "Mini-Imagenet(2400 samples)", fontsize=fontsize+10)




tes = tes_all['food1k']
btes = btes_all['food1k']
ftes = ftes_all['food1k']
df = df_all['food1k']
te = te_scatter['food1k']

task_num = 50
total_alg = 4
combined_alg_name = ['SynN','SynF', 'Model Zoo', 'LwF']

marker_style = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']
marker_style_scatter = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']

clr_combined = ["#377eb8", "#e41a1c", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928"]
c_combined = sns.color_palette(clr_combined, n_colors=total_alg)

fontsize=29
ticksize=26
legendsize=14

ax = fig.add_subplot(gs[34:42,:8])

for i, fte in enumerate(ftes):
    fte[0] = 1
    if i == 0:
        ax.plot(np.arange(1,51), fte, color=c_combined[i], marker=marker_style[i], markersize=12, label=combined_alg_name[i], linewidth=3)
        continue

    if i == 1:
        ax.plot(np.arange(1,51), fte, color=c_combined[i], marker=marker_style[i], markersize=12, label=combined_alg_name[i], linewidth=3)
        continue
    
    ax.plot(np.arange(1,51), fte, color=c_combined[i], marker=marker_style[i], markersize=12, label=combined_alg_name[i])

ax.set_title('Forward Learning (FL)', fontsize=fontsize+5)
ax.set_xticks([1,10,20,30,40,50])
ax.set_yticks([0.9,1,1.5])
#ax.set_yticks([])
#ax.text(0, np.mean(ax.get_ylim()), "%s" % str(0), fontsize=26)
#ax.yaxis.set_major_locator(plt.LogLocator(subs=(0.9, 1, 1.1, 1.2, 1.3)))
#ax.set_ylim(0.89, 1.31)

log_lbl = np.round(
    np.log([0.9,1,1.5]),
    1
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)

ax.tick_params(labelsize=ticksize)

ax.set_ylabel('log FLE', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,51, colors='grey', linestyles='dashed',linewidth=1.5)

handles, labels_ = ax.get_legend_handles_labels()

ax = fig.add_subplot(gs[32:45,8:27], projection='3d')

xs = np.linspace(0, task_num, 100)
for ii in range(total_alg):
    zs = np.linspace(ii-.005,ii+.005,10)
    X, Y = np.meshgrid(xs, zs)
    Z = np.ones(X.shape)

    ax.plot_surface(X, Y, Z, color='grey', alpha=1)

#cmap = sns.color_palette("coolwarm", as_cmap=True)
color = ['b', 'r']
for i in range(task_num - 1):

    et = np.zeros((total_alg,task_num-i))

    for j in range(0,total_alg):
        et[j,:] = np.asarray(btes[j][i])

    ns = np.arange(i + 1, task_num + 1)
    ns_new = np.linspace(ns.min(), ns.max(), 300)

    for j in range(0,total_alg):
        y_interp = np.interp(ns_new, ns, et[j,:])
        idx = np.where(y_interp < 1.0)[0]
        supper = y_interp.copy()
        supper[idx] = np.nan

        idx = np.where(y_interp >= 1.0)[0]
        slower = y_interp.copy()
        slower[idx] = np.nan

        ax.plot(ns_new, supper, zs=j, zdir='y', color='r', linewidth=3)
        ax.plot(ns_new, slower, zs=j, zdir='y', color='b', linewidth=3)
    

zs = np.linspace(0, total_alg-1, 100)
X, Y = np.meshgrid(xs, zs)
Z = np.ones(X.shape)

ax.plot_surface(X, Y, Z, color='grey', alpha=.3)

ax.view_init(elev=10., azim=15, roll=0)

'''for i in range(total_alg_top,total_alg_top+total_alg_bottom-1):
    ax.plot(1,0,color=c_combined[i], marker=marker_style[i], markersize=8,label=combined_alg_name[i])'''


ax.text(.9, .5, 1.7, 'Backward Learning (BL)', fontsize=fontsize+5)
ax.text(.1, .1, 1.8, "FOOD1k 50X20 (1200 samples)", fontsize=fontsize+20)
ax.set_xlabel('Tasks seen', fontsize=30, labelpad=15)
ax.set_zlabel('log BLE', fontsize=30, labelpad=15)

ax.set_zticks([.7,1,1.5])
ax.set_xticks([1,20,50])
ax.set_yticks(np.arange(0,total_alg,1))
ax.set_zlim(0.7, 1.5)
ax.set_ylim([0,3])
log_lbl = np.round(
    np.log([.7,1,1.5]),
    1
)
labels = [item.get_text() for item in ax.get_zticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_zticklabels(labels)
ax.set_yticklabels(combined_alg_name, rotation=80)
ax.tick_params(labelsize=ticksize-4)
#ax[0][1].grid(axis='x')
ax.invert_xaxis()


#ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

for ytick, color in zip(ax.get_yticklabels(), clr_combined):
    ytick.set_color(color)

###############################################

ax = fig.add_subplot(gs[34:42,29:37])
ax.tick_params(labelsize=22)
ax_ = sns.boxplot(
    x="Algorithms", y="Transfer Efficieny", data=df, palette=c_combined, whis=np.inf,
    ax=ax, showfliers=False, notch=1
    )
ax.hlines(0, -1,4, colors='grey', linestyles='dashed',linewidth=1.5)
#sns.boxplot(x="Algorithms", y="Transfer Efficieny", data=mean_df, palette=c, linewidth=3, ax=ax[1][1])
#ax_=sns.pointplot(x="Algorithms", y="Transfer Efficieny", data=df_500, join=False, color='grey', linewidth=1.5, ci='sd',ax=ax)
ax_.set_yticks([-.3,0,.2,.4])
ax.set_title('Overall Learning', fontsize=fontsize+5)
ax_.set_xlabel('', fontsize=fontsize)
ax.set_ylabel('log LE after 50 Tasks', fontsize=fontsize-5)
ax_.set_xticklabels(
    ['SynN','SynF', 'Model zoo', 'LwF'],
    fontsize=22,rotation=65,ha="right",rotation_mode='anchor'
    )

stratified_scatter(te,ax,16,c_combined,marker_style_scatter)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig('combined.pdf')
# %%