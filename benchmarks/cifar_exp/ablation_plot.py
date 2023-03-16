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
shifts = 6
total_alg_top = 4
total_alg_bottom = 4
alg_name_top = ['SynN (.4)', 'SynN (.6)', 'SynN (.8)', 'SynN (1)']
alg_name_bottom = ['SynF (.4)', 'SynF (.6)', 'SynF (.8)', 'SynF (1)']
combined_alg_name = ['SynN (.4)', 'SynN (.6)', 'SynN (.8)', 'SynN (1)', 'SynF (.4)', 'SynF (.6)', 'SynF (.8)', 'SynF (1)']
model_file_top = ['dnn0']
model_file_bottom = ['uf10']
samples_to_replay = [.4,.6,.8,1]

btes_top = [[] for i in range(total_alg_top)]
ftes_top = [[] for i in range(total_alg_top)]
tes_top = [[] for i in range(total_alg_top)]
btes_bottom = [[] for i in range(total_alg_bottom)]
ftes_bottom = [[] for i in range(total_alg_bottom)]
tes_bottom = [[] for i in range(total_alg_bottom)]
avg_acc_top = [[] for i in range(total_alg_top)]
avg_var_top = [[] for i in range(total_alg_top)]
avg_acc_bottom = [[] for i in range(total_alg_bottom)]
avg_var_bottom = [[] for i in range(total_alg_bottom)]

avg_single_acc_top = [[] for i in range(total_alg_top)]
avg_single_var_top = [[] for i in range(total_alg_top)]
avg_single_acc_bottom = [[] for i in range(total_alg_bottom)]
avg_single_var_bottom = [[] for i in range(total_alg_bottom)]

#combined_alg_name = ['L2N','L2F','Prog-NN', 'DF-CNN','LwF','EWC','O-EWC','SI', 'Replay (increasing amount)', 'Replay (fixed amount)', 'None']
model_file_combined = ['dnn0','uf10']

########################

#%% code for 500 samples
reps = slots*shifts

for alg, samples in enumerate(samples_to_replay):

    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for slot in range(slots):
        for shift in range(shifts):
            filename = './result/result/'+model_file_top[0]+'_'+str(shift+1)+'_'+str(slot)+'_'+str(samples)+'.pickle'

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
    avg_acc, avg_var = calc_avg_acc(err, reps)
    avg_single_acc, avg_single_var = calc_avg_single_acc(single_err, reps)

    btes_top[alg].extend(bte)
    ftes_top[alg].extend(fte)
    tes_top[alg].extend(te)
    avg_acc_top[alg]= avg_acc
    avg_var_top[alg] = avg_var
    avg_single_acc_top[alg]= avg_single_acc
    avg_single_var_top[alg] = avg_single_var

    print('Algo name:' , alg_name_top[0], 'replay ', samples)
    print('Accuracy', np.round(calc_acc(err,reps),2))
    print('forget', np.round(calc_forget(err, reps),2))
    print('transfer', np.round(calc_transfer(err, single_err, reps),2))
    

# %%
reps = slots*shifts

for alg, samples in enumerate(samples_to_replay):
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for slot in range(slots):
        for shift in range(shifts):
            filename = './result/result/'+model_file_bottom[0]+'_'+str(shift+1)+'_'+str(slot)+'_'+str(samples)+'.pickle'
            
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
    avg_acc, avg_var = calc_avg_acc(err, reps)
    avg_single_acc, avg_single_var = calc_avg_single_acc(single_err, reps)


    btes_bottom[alg].extend(bte)
    ftes_bottom[alg].extend(fte)
    tes_bottom[alg].extend(te)
    avg_acc_bottom[alg] = avg_acc
    avg_var_bottom[alg] = avg_var
    avg_single_acc_bottom[alg]= avg_single_acc
    avg_single_var_bottom[alg] = avg_single_var

    print('Algo name:' , alg_name_bottom[0], 'replay', samples)
    print('Accuracy', np.round(calc_acc(err,reps),2))
    print('forget', np.round(calc_forget(err, reps),2))
    print('transfer', np.round(calc_transfer(err, single_err, reps),2))
#%%
te_500 = {'SynN (.4)':np.zeros(10,dtype=float), 'SynN (.6)':np.zeros(10,dtype=float), 
          'SynN (.8)':np.zeros(10,dtype=float),
          'SynN (1)':np.zeros(10,dtype=float), 'SynF (.4)':np.zeros(10,dtype=float), 
          'SynF (.6)':np.zeros(10,dtype=float),'SynF (.8)':np.zeros(10,dtype=float),
          'SynF (1)':np.zeros(10,dtype=float)}

for count,name in enumerate(te_500.keys()):
    print(name, count)
    for i in range(10):
        if count <4:
            te_500[name][i] = np.log(tes_top[count][i][9-i])
        else:
            te_500[name][i] = np.log(tes_bottom[count-8][i][9-i])


df_500 = pd.DataFrame.from_dict(te_500)
df_500 = pd.melt(df_500,var_name='Algorithms', value_name='Learning Efficieny')

# %%
fig = plt.figure(constrained_layout=True,figsize=(38,14))
gs = fig.add_gridspec(14,38)

clr = ["#984ea3", "#984ea3", "#984ea3", "#984ea3", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a"]

fontsize=38
ticksize=34
legendsize=16

ax = fig.add_subplot(gs[4:12,:8])

fte = ftes_top[3]
fte[0] = 1
ax.plot(np.arange(1,11), fte, color=clr[0], label='SynN', linewidth=3)

fte = ftes_bottom[3]
fte[0] = 1
ax.plot(np.arange(1,11), fte, color=clr[4], label='SynF', linewidth=3)

ax.set_title('Forward Learning (FL)', fontsize=fontsize)
ax.set_xticks(np.arange(1,11))
ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3])
#ax.set_yticks([])
#ax.text(0, np.mean(ax.get_ylim()), "%s" % str(0), fontsize=26)
#ax.yaxis.set_major_locator(plt.LogLocator(subs=(0.9, 1, 1.1, 1.2, 1.3)))
ax.set_ylim(0.9, 1.31)

log_lbl = np.round(
    np.log([0.9,1,1.1,1.2,1.3]),
    1
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)

ax.tick_params(labelsize=ticksize)

ax.set_ylabel('log FLE', fontsize=fontsize)
ax.set_xlabel('Tasks seen', fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

handles_top, labels_top = ax.get_legend_handles_labels()


#########################################################
#ax[0][0].grid(axis='x')
ax = fig.add_subplot(gs[2:15,8:27], projection='3d')
#cmap = sns.color_palette("coolwarm", as_cmap=True)
color = ['b', 'r']
for i in range(task_num - 1):

    et = np.zeros((total_alg_top,task_num-i))

    for j in range(0,total_alg_top):
        et[j,:] = np.asarray(btes_top[j][i])

    ns = np.arange(i + 1, task_num + 1)
    ns_new = np.linspace(ns.min(), ns.max(), 60)

    for j in range(0,total_alg_top):
        y_interp = np.interp(ns_new, ns, et[j,:])
        idx = np.where(y_interp < 1.0)[0]
        supper = y_interp.copy()
        supper[idx] = np.nan

        idx = np.where(y_interp >= 1.0)[0]
        slower = y_interp.copy()
        slower[idx] = np.nan

        ax.plot(ns_new, supper, zs=j, zdir='y', color='r', linewidth=3)
        ax.plot(ns_new, slower, zs=j, zdir='y', color='b', linewidth=3)
        



for i in range(task_num - 1):

    et = np.zeros((total_alg_bottom,task_num-i))

    for j in range(0,total_alg_bottom):
        et[j,:] = np.asarray(btes_bottom[j][i])

    ns = np.arange(i + 1, task_num + 1)
    ns_new = np.linspace(ns.min(), ns.max(), 60)

    for j in range(0,total_alg_bottom):
        y_interp = np.interp(ns_new, ns, et[j,:])
        idx = np.where(y_interp < 1.0)[0]
        supper = y_interp.copy()
        supper[idx] = np.nan

        idx = np.where(y_interp >= 1.0)[0]
        slower = y_interp.copy()
        slower[idx] = np.nan

        ax.plot(ns_new, supper, zs=j+4, zdir='y', color='r', linewidth=3)
        ax.plot(ns_new, slower, zs=j+4, zdir='y', color='b', linewidth=3)
        

xs = np.linspace(0, 10, 10)
zs = np.linspace(0, 7, 10)
X, Y = np.meshgrid(xs, zs)
Z = np.ones(X.shape)

ax.plot_surface(X, Y, Z, color='grey', alpha=.3)

for ii in range(total_alg_top):
    zs = np.linspace(ii-.05,ii+.05,10)
    X, Y = np.meshgrid(xs, zs)
    Z = np.ones(X.shape)

    ax.plot_surface(X, Y, Z, color='grey', alpha=1)

for ii in range(4,4+total_alg_bottom):
    zs = np.linspace(ii-.05,ii+.05,10)
    X, Y = np.meshgrid(xs, zs)
    Z = np.ones(X.shape)

    ax.plot_surface(X, Y, Z, color='grey', alpha=1)


ax.view_init(elev=10., azim=15, roll=0)

'''for i in range(total_alg_top,total_alg_top+total_alg_bottom-1):
    ax.plot(1,0,color=c_combined[i], marker=marker_style[i], markersize=8,label=combined_alg_name[i])'''

ax.text(.9, .5, 1.4, 'Backward Learning (BL)', fontsize=fontsize)
ax.set_xlabel('Tasks seen', fontsize=30, labelpad=15)
ax.set_zlabel('log BLE', fontsize=30, labelpad=15)

ax.set_zticks([.9,1, 1.1,1.2])
ax.set_xticks(np.arange(2,11,4))
ax.set_zlim(0.9, 1.25)
ax.set_ylim([0,7])
log_lbl = np.round(
    np.log([.9,1,1.1,1.2]),
    1
)
labels = [item.get_text() for item in ax.get_zticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_zticklabels(labels)
ax.set_yticks(np.arange(0,8,1))
ax.set_yticklabels(combined_alg_name, rotation=80)
ax.tick_params(labelsize=ticksize-8)
#ax[0][1].grid(axis='x')
ax.invert_xaxis()

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
#ax.hlines(1, 1,10, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

for ytick, color in zip(ax.get_yticklabels(), clr):
    ytick.set_color(color)


##########################################################
ax = fig.add_subplot(gs[4:12,26:34])

ax.set_title('Overall Learning', fontsize=fontsize)
ax.tick_params(labelsize=22)
ax_ = sns.boxplot(
    x="Algorithms", y="Learning Efficieny", data=df_500, palette=clr, whis=np.inf,
    ax=ax, showfliers=False, notch=1
    )
ax.hlines(0, -1,8, colors='grey', linestyles='dashed',linewidth=1.5, label='chance')
#sns.boxplot(x="Algorithms", y="Transfer Efficieny", data=mean_df, palette=c, linewidth=3, ax=ax[1][1])
#ax_=sns.pointplot(x="Algorithms", y="Transfer Efficieny", data=df_500, join=False, color='grey', linewidth=1.5, ci='sd',ax=ax)
#ax_.set_yticks([.4,.6,.8,1, 1.2,1.4])
ax_.set_xlabel('', fontsize=fontsize)
ax.set_ylabel('log LE after 10 Tasks', fontsize=fontsize-5)
ax_.set_xticklabels(
    ['SynN (0.4)','SynN (0.6)', 'SynN (0.8)', 'SynN (1)','SynF (0.4)', 'SynF (0.6)', 'SynF (0.8)', 'SynF (1)'],
    fontsize=22,rotation=80,ha="right",rotation_mode='anchor'
    )

stratified_scatter(te_500,ax,16,clr)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

#########################################################

fig.text(.35, 0.88, "CIFAR 10X10 (Ablation)", fontsize=fontsize+10)
fig.legend(handles_top, labels_top, bbox_to_anchor=(.9, .5), fontsize=legendsize+14, frameon=False)
#fig.legend(handles_bottom, labels_bottom, bbox_to_anchor=(1, .45), fontsize=legendsize+14, frameon=False)

plt.savefig('result/figs/ablation.pdf', dpi=300)


# %%
