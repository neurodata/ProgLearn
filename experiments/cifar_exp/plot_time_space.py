#%%
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
from itertools import product
import seaborn as sns

### MAIN HYPERPARAMS ###
ntrees = 10
slots = 1
shifts = 6
task_num = 10
model = "uf"
########################

#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_fte_bte(err, single_err, ntrees):
    bte = [[] for i in range(10)]
    te = [[] for i in range(10)]
    fte = []
    
    for i in range(10):
        for j in range(i,10):
            #print(err[j][i],j,i)
            bte[i].append(err[i][i]/err[j][i])
            te[i].append(single_err[i]/err[j][i])
                
    for i in range(10):
        #print(single_err[i],err[i][i])
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

def calc_mean_err(err,task_num=10,reps=6):
    mean_err = [[] for i in range(task_num)]


    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(err[i][j])
        
        tmp=tmp/reps
        #print(tmp)
        mean_err[j].extend([tmp])
            
    return mean_err 

def calc_mean_multitask_time(multitask_time,task_num=10,reps=6):
    mean_multitask_time = [[] for i in range(task_num)]


    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(multitask_time[i][j])
        
        tmp=tmp/reps
        #print(tmp)
        mean_multitask_time[j].extend([tmp])
            
    return mean_multitask_time

def calc_mean_multitask_space(multitask_space,task_num=10,reps=6):
    mean_multitask_space = [[] for i in range(task_num)]


    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(multitask_space[i][j])
        
        tmp=tmp/reps
        #print(tmp)
        mean_multitask_space[j].extend([tmp])
            
    return mean_multitask_space

#%%
reps = slots*shifts

btes = [[] for i in range(task_num)]
ftes = [[] for i in range(task_num)]
tes = [[] for i in range(task_num)]
err_ = [[] for i in range(task_num)]

te_tmp = [[] for _ in range(reps)]
bte_tmp = [[] for _ in range(reps)]
fte_tmp = [[] for _ in range(reps)]
err_tmp = [[] for _ in range(reps)]
train_time_tmp = [[] for _ in range(reps)]
single_task_inference_time_tmp = [[] for _ in range(reps)]
multitask_inference_time_tmp = [[] for _ in range(reps)]
multitask_inference_space_tmp = [[] for _ in range(reps)]

count = 0   
for slot in range(slots):
    for shift in range(shifts):
        filename = 'result/result/reduced_sample_'+model+str(ntrees)+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
        multitask_df, single_task_df = unpickle(filename)

        err = [[] for _ in range(10)]
        multitask_inference_times = [[] for _ in range(10)]

        for ii in range(10):
            err[ii].extend(
             1 - np.array(
                 multitask_df[multitask_df['base_task']==ii+1]['accuracy']
             )
            )
            multitask_inference_times[ii].extend(np.array(multitask_df[multitask_df['base_task']==ii+1]['multitask_inference_times']))
        
        
        single_err = 1 - np.array(single_task_df['accuracy'])
        fte, bte, te = get_fte_bte(err,single_err,ntrees)
    
        err_ = [[] for i in range(task_num)]
        for i in range(task_num):
            for j in range(task_num-i):
                #print(err[i+j][i])
                err_[i].append(err[i+j][i])
           
        train_time_tmp[count].extend(np.array(single_task_df['train_times']))
        single_task_inference_time_tmp[count].extend(np.array(single_task_df['single_task_inference_times']))
        multitask_inference_time_tmp[count].extend(multitask_inference_times)
        multitask_inference_space_tmp[count].extend(np.array(single_task_df['model_size'])/1024)
        te_tmp[count].extend(te)
        bte_tmp[count].extend(bte)
        fte_tmp[count].extend(fte)
        err_tmp[count].extend(err_)
        count+=1
    
te = calc_mean_te(te_tmp,reps=reps)
bte = calc_mean_bte(bte_tmp,reps=reps)
fte = calc_mean_fte(fte_tmp,reps=reps)
error = calc_mean_err(err_tmp,reps=reps)

train_time = np.mean(train_time_tmp, axis = 0)
single_task_inference_time = np.mean(single_task_inference_time_tmp, axis = 0)
multitask_inference_time = calc_mean_multitask_time(multitask_inference_time_tmp)
multitask_inference_time = [np.mean(multitask_inference_time[i]) for i in range(len(multitask_inference_time))]
multitask_inference_space = calc_mean_multitask_space(multitask_inference_space_tmp)

#%%
btes = [[] for i in range(task_num)]
ftes = [[] for i in range(task_num)]
tes = [[] for i in range(task_num)]
err_ = [[] for i in range(task_num)]

te_tmp = [[] for _ in range(reps)]
bte_tmp = [[] for _ in range(reps)]
fte_tmp = [[] for _ in range(reps)]
err_tmp = [[] for _ in range(reps)]
train_time_tmp = [[] for _ in range(reps)]
single_task_inference_time_tmp = [[] for _ in range(reps)]
multitask_inference_time_tmp = [[] for _ in range(reps)]
multitask_inference_space_tmp = [[] for _ in range(reps)]

count = 0   
for slot in range(slots):
    for shift in range(shifts):
        filename = 'result/result/reduced_sample_dnn0'+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
        multitask_df, single_task_df = unpickle(filename)

        err = [[] for _ in range(10)]
        multitask_inference_times = [[] for _ in range(10)]

        for ii in range(10):
            err[ii].extend(
             1 - np.array(
                 multitask_df[multitask_df['base_task']==ii+1]['accuracy']
             )
            )
            multitask_inference_times[ii].extend(np.array(multitask_df[multitask_df['base_task']==ii+1]['multitask_inference_times']))
        
        
        single_err = 1 - np.array(single_task_df['accuracy'])
        fte, bte, te = get_fte_bte(err,single_err,ntrees)
    
        err_ = [[] for i in range(task_num)]
        for i in range(task_num):
            for j in range(task_num-i):
                #print(err[i+j][i])
                err_[i].append(err[i+j][i])
           
        train_time_tmp[count].extend(np.array(single_task_df['train_times']))
        single_task_inference_time_tmp[count].extend(np.array(single_task_df['single_task_inference_times']))
        multitask_inference_time_tmp[count].extend(multitask_inference_times)
        multitask_inference_space_tmp[count].extend(np.array(single_task_df['model_size'])/1024)
        te_tmp[count].extend(te)
        bte_tmp[count].extend(bte)
        fte_tmp[count].extend(fte)
        err_tmp[count].extend(err_)
        count+=1
    
te_ = calc_mean_te(te_tmp,reps=reps)
bte_ = calc_mean_bte(bte_tmp,reps=reps)
fte_ = calc_mean_fte(fte_tmp,reps=reps)
error_ = calc_mean_err(err_tmp,reps=reps)

train_time_= np.mean(train_time_tmp, axis = 0)
single_task_inference_time_ = np.mean(single_task_inference_time_tmp, axis = 0)
multitask_inference_time_ = calc_mean_multitask_time(multitask_inference_time_tmp)
multitask_inference_time_ = [np.mean(multitask_inference_time[i]) for i in range(len(multitask_inference_time))]
multitask_inference_space_ = calc_mean_multitask_space(multitask_inference_space_tmp)

#%%
sns.set_context("talk")

n_tasks=10
clr = ["#e41a1c", "#a65628", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#CCCC00"]
#c = sns.color_palette(clr, n_colors=len(clr))

fontsize=22
ticksize=20

fig, ax = plt.subplots(3,2, figsize=(24,15))
#fig.suptitle('ntrees = '+str(ntrees),fontsize=25)
ax[0][0].plot(np.arange(1,n_tasks+1), fte, label='L2F', c='red', marker='.', markersize=14, linewidth=3)
ax[0][0].plot(np.arange(1,n_tasks+1), fte_, label='L2N', c='blue', marker='.', markersize=14, linewidth=3)
ax[0][0].hlines(1, 1,n_tasks, colors='grey', linestyles='dashed',linewidth=1.5)
ax[0][0].tick_params(labelsize=ticksize)
ax[0][0].set_xlabel('Number of tasks seen', fontsize=fontsize)
ax[0][0].set_ylabel('FTE', fontsize=fontsize)
ax[0][0].legend(fontsize=22)

for i in range(n_tasks):

    et = np.asarray(bte[i])
    et_ = np.asarray(bte_[i])

    ns = np.arange(i + 1, n_tasks + 1)
    ax[0][1].plot(ns, et, c='red', label='L2F', linewidth = 2.6)
    ax[0][1].plot(ns, et_, c='blue', label='L2N', linewidth = 2.6)
    
ax[0][1].set_xlabel('Number of tasks seen', fontsize=fontsize)
ax[0][1].set_ylabel('BTE', fontsize=fontsize)
#ax[0][1].set_xticks(np.arange(1,10))
ax[0][1].tick_params(labelsize=ticksize)
ax[0][1].hlines(1, 1,n_tasks, colors='grey', linestyles='dashed',linewidth=1.5)


for i in range(n_tasks):

    et = np.asarray(te[i])
    et_ = np.asarray(te_[i])

    ns = np.arange(i + 1, n_tasks + 1)
    ax[1][0].plot(ns, et, c='red', linewidth = 2.6)
    ax[1][0].plot(ns, et_, c='blue', linewidth = 2.6)
    
ax[1][0].set_xlabel('Number of tasks seen', fontsize=fontsize)
ax[1][0].set_ylabel('Transfer Efficiency', fontsize=fontsize)
#ax[1][0].set_xticks(np.arange(1,10))
ax[1][0].tick_params(labelsize=ticksize)
ax[1][0].hlines(1, 1,n_tasks, colors='grey', linestyles='dashed',linewidth=1.5)

'''for rep in range(reps):
    _, single_task_df = unpickle('./result/'+model+str(ntrees)+'__'+str(rep+1)+'.pickle')
    single_err = 1 - np.array(single_task_df['accuracy'])
   
    for i in range(n_tasks):
        et = np.asarray(err_tmp[rep][i])
        ns = np.arange(i + 1, n_tasks + 1)
        
        ax[1][1].plot(i+1, 1-single_err[i], marker='o',c=c[rep])
        if i==0:
            ax[1][1].plot(ns, 1-et, c=c[rep], label='rep '+str(rep+1) ,linewidth = 2.6)
        else:
            ax[1][1].plot(ns, 1-et, c=c[rep], linewidth = 2.6)
'''

for i in range(n_tasks):
    et = np.asarray(error[i][0])
    et_ = np.asarray(error_[i][0])
    ns = np.arange(i + 1, n_tasks + 1)

    ax[1][1].plot(ns, 1-et , c='red', linewidth = 2.6)
    ax[1][1].plot(ns, 1-et_ , c='blue', linewidth = 2.6)
            
#ax[1][1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=22)
ax[1][1].set_xlabel('Number of tasks seen', fontsize=fontsize)
ax[1][1].set_ylabel('Accuracy', fontsize=fontsize)
#ax[1][1].set_yticks([.4,.6,.8,.9,1, 1.1,1.2])
#ax[1][1].set_xticks(np.arange(1,10))
#ax[1][1].set_ylim(0.89, 1.15)
ax[1][1].tick_params(labelsize=ticksize)

ax[2][0].plot(range(len(multitask_inference_time)), multitask_inference_time, c='red', linewidth=3, linestyle="solid", label = "Multi-Task Inference Time")
ax[2][0].plot(range(len(multitask_inference_time_)), multitask_inference_time_, c='blue', linewidth=3, linestyle="solid", label = "Multi-Task Inference Time")

ax[2][0].set_xlabel('Number of Tasks Seen', fontsize=fontsize)
ax[2][0].set_ylabel('Time (seconds)', fontsize=fontsize)
ax[2][0].tick_params(labelsize=ticksize)
#plt.savefig('./result/figs/fig_trees'+str(ntrees)+"__"+model+'.pdf',dpi=300)
#plt.close()

ax[2][1].plot(range(len(multitask_inference_space)), multitask_inference_space, c= 'red', linewidth=3, linestyle="solid", label = "Multi-Task Inference Time")
ax[2][1].plot(range(len(multitask_inference_space_)), multitask_inference_space_, c= 'blue', linewidth=3, linestyle="solid", label = "Multi-Task Inference Time")

ax[2][1].set_xlabel('Number of Tasks Seen', fontsize=fontsize)
ax[2][1].set_ylabel('KB', fontsize=fontsize)
ax[2][1].tick_params(labelsize=ticksize)

plt.savefig('./result/figs/space_time_efficiency.pdf')
# %%
