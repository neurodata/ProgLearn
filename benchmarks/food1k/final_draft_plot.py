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
def calc_forget(err, reps, total_task=50):
#Tom Vient et al
    forget = 0
    for ii in range(total_task-1):
        forget += err[ii][ii] - err[total_task-1][ii]

    forget /= (total_task-1)
    return forget/reps

def calc_transfer(err, single_err, reps, total_task=50):
#Tom Vient et al
    transfer = np.zeros(total_task,dtype=float)

    for ii in range(total_task):
        transfer[ii] = (single_err[ii] - err[total_task-1][ii])/reps

    return np.mean(transfer)

def calc_acc(err, reps, total_task=50):
#Tom Vient et al
    acc = 0
    for ii in range(total_task):
        acc += (1-err[total_task-1][ii]/reps)
    return acc/total_task


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def calc_avg_acc(err, reps):
    avg_acc = np.zeros(50, dtype=float)
    avg_var = np.zeros(50, dtype=float)
    for i in range(50):
        avg_acc[i] = (1*(i+1) - np.sum(err[i])/reps + (4-i)*.1)/20
        avg_var[i] = np.var(1-np.array(err[i])/reps)
    return avg_acc, avg_var

def calc_avg_single_acc(err, reps):
    avg_acc = np.zeros(50, dtype=float)
    avg_var = np.zeros(50, dtype=float)
    for i in range(50):
        avg_acc[i] = (1*(i+1) - np.sum(err[:i+1])/reps + (4-i)*.1)/20
        avg_var[i] = np.var(1-np.array(err[:i+1])/reps)
    return avg_acc, avg_var
    
def get_fte_bte(err, single_err):
    bte = [[] for i in range(50)]
    te = [[] for i in range(50)]
    fte = []
    
    for i in range(50):
        for j in range(i,50):
            #print(err[j][i],j,i)
            #print(len(bte), bte[i])
            bte[i].append(err[i][i]/err[j][i])
            te[i].append(single_err[i]/err[j][i])
                
    for i in range(50):
        fte.append(single_err[i]/err[i][i])
            
            
    return fte,bte,te

def calc_mean_bte(btes,task_num=50,reps=1):
    mean_bte = [[] for i in range(task_num)]


    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(btes[i][j])
        
        tmp=tmp/reps
        mean_bte[j].extend(tmp)
            
    return mean_bte     

def calc_mean_te(tes,task_num=50,reps=1):
    mean_te = [[] for i in range(task_num)]

    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(tes[i][j])
        
        tmp=tmp/reps
        mean_te[j].extend(tmp)
                                             
    return mean_te 

def calc_mean_fte(ftes,task_num=50,reps=1):
    fte = np.asarray(ftes)
    
    return list(np.mean(np.asarray(fte),axis=0))

def get_error_matrix(filename):
    multitask_df, single_task_df = unpickle(filename)

    err = [[] for _ in range(50)]

    for ii in range(50):
        err[ii].extend(
            1 - np.array(
                multitask_df[multitask_df['base_task']==ii+1]['accuracy']
                )
            )
    single_err = 1 - np.array(single_task_df['accuracy'])

    return single_err, err

def sum_error_matrix(error_mat1, error_mat2):
    err = [[] for _ in range(50)]

    for ii in range(20):
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

    filename = './results/'+model_file_combined[alg]+'.pickle'

    multitask_df, single_task_df = unpickle(filename)

    single_err, err = get_error_matrix(filename)

    #single_err /= reps
    #err /= reps
    fte, bte, te = get_fte_bte(err,single_err)
    avg_acc_, avg_var_ = calc_avg_acc(err, 1)
    avg_single_acc_, avg_single_var_ = calc_avg_single_acc(single_err, 1)

    btes[alg].extend(bte)
    ftes[alg].extend(fte)
    tes[alg].extend(te)
    avg_acc[alg]= avg_acc_
    avg_var[alg] = avg_var_
    avg_single_acc[alg]= avg_single_acc_
    avg_single_var[alg] = avg_single_var_

    print('Algo name:' , combined_alg_name[alg])
    print('Accuracy', np.round(calc_acc(err,1),2))
    print('forget', np.round(calc_forget(err, 1),2))
    print('transfer', np.round(calc_transfer(err, single_err, 1),2))

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

# %%
fig = plt.figure(constrained_layout=True,figsize=(40,12))
gs = fig.add_gridspec(12, 40)

marker_style = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']
marker_style_scatter = ['.', '.', 'v', '.', '+', 'o', '*', 'o', '*', 'x', '.', '+', 'v']

clr_combined = ["#377eb8", "#e41a1c", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928"]
c_combined = sns.color_palette(clr_combined, n_colors=total_alg)

fontsize=29
ticksize=26
legendsize=14

ax = fig.add_subplot(gs[2:10,1:9])

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

ax = fig.add_subplot(gs[:13,9:28], projection='3d')

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

ax = fig.add_subplot(gs[2:10,29:37])
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
    fontsize=18,rotation=65,ha="right",rotation_mode='anchor'
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
        ax.plot(np.arange(1,51,1) ,avg_acc[i], color=c_combined[i], marker=marker_style[i], linewidth=3)
    else:
        ax.plot(np.arange(1,51,1) ,avg_acc[i], color=c_combined[i], marker=marker_style[i])
    #ax.fill_between(np.arange(1,6,1), avg_acc[i]-1.96*avg_var[i], avg_acc[i]+1.96*avg_var[i], facecolor=c_combined[i], alpha=.3)

ax.hlines(.05, 1,51, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')
ax.set_yticks([.2,.4,.6,.8,1])
ax.set_xticks([1,10,20,30,40,50])
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
        ax.plot(np.arange(1,51,1) ,avg_single_acc[i], color=c_combined[i], marker=marker_style[i], linewidth=3)
    else:
        ax.plot(np.arange(1,51,1) ,avg_single_acc[i], color=c_combined[i], marker=marker_style[i])
   # ax.fill_between(np.arange(1,6,1), avg_single_acc[i]-1.96*avg_single_var[i], avg_single_acc[i]+1.96*avg_single_var[i], facecolor=c_combined[i], alpha=.3)

ax.hlines(.05, 1,51, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')
ax.set_yticks([.2,.4,.6,.8,1])
ax.set_xticks([1,10,20,30,40,50])
ax.tick_params(labelsize=ticksize)
ax.set_ylabel('Single task accuracy', fontsize=fontsize)
ax.set_xlabel('Number of tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)'''
fig.text(.38, 0.88, "FOOD1k (1200 samples)", fontsize=fontsize+15)

plt.savefig('food1k.pdf')
# %%
