#%%
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
from itertools import product
import seaborn as sns

### MAIN HYPERPARAMS ###
slots = 10
shifts = 6
alg_name = ['L2N','L2F','Prog_NN', 'DF_CNN','LwF','EWC','Online_EWC','SI']
########################

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

def calc_mean_bte_(btes,task_num=10,reps=6):
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


def get_bte(err):
    bte = []
    
    for i in range(10):
        bte.append(err[0] / err[i])
    
    return bte

def calc_mean_bte(btes,task_num=10,reps=6):
    mean_bte = []
    

    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            print(np.array(btes[i]))
            tmp += np.array(btes[i])
        
        tmp=tmp/reps
        mean_bte.extend(tmp)
            
    return mean_bte       

#%% without shuffling result
reps = slots*shifts
btes_org = np.zeros((len(alg_name),10),dtype=float)
total_alg = 8

for alg_no,alg in enumerate(alg_name):
    count = 0 
    bte_tmp = [[] for _ in range(reps)]

    for slot in range(slots):
        for shift in range(shifts):
            if alg_no==0:
                filename = '../cifar_exp/result/result/dnn0_'+str(shift+1)+'_'+str(slot)+'.pickle'
            elif alg_no==1:
                filename = '../cifar_exp/result/result/uf10_'+str(shift+1)+'_'+str(slot)+'.pickle'
            else:
                filename = '../cifar_exp/benchmarking_algorthms_result/'+alg+'_'+str(shift+1)+'_'+str(slot)+'.pickle'

            multitask_df, single_task_df = unpickle(filename)

            single_err, err = get_error_matrix(filename)
            _, bte, _ = get_fte_bte(err,single_err)
            
            bte_tmp[count].extend(bte)
            count+=1
    
    btes_org[alg_no][:] = np.array(calc_mean_bte_(bte_tmp,reps=reps)[0])

#%% shuffling result
reps = slots*shifts
btes = np.zeros((len(alg_name),10),dtype=float)

for alg_no,alg in enumerate(alg_name):
    bte_tmp = [[] for _ in range(reps)]

    count = 0   
    for slot in range(slots):
        for shift in range(shifts):
            if alg_no==0:
                filename = 'result/dnn0_'+str(shift+1)+'_'+str(slot)+'.pickle'
            elif alg_no==1:
                filename = 'result/uf10_'+str(shift+1)+'_'+str(slot)+'.pickle'
            else:
                filename = 'benchmarking_algorthms_result/'+alg+'_'+str(shift+1)+'_'+str(slot)+'.pickle'

            multitask_df = unpickle(filename)

            err = []

            for ii in range(10):
                err.extend(
                1 - np.array(
                    multitask_df[multitask_df['task']==ii+1]['task_1_accuracy']
                )
                )
            bte = get_bte(err)
        
            bte_tmp[count].extend(bte)
            count+=1
    
    btes[alg_no] = np.mean(bte_tmp, axis = 0)
    
summary = (btes_org,btes)
with open('../plot_label_shuffled_angle_recruitment/label_shuffle_result/res.pickle','wb') as f:
    pickle.dump(summary,f)

#%%
clr = ["#00008B", "#e41a1c", "#a65628", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#CCCC00"]
c = sns.color_palette(clr, n_colors=len(clr))
fig, ax = plt.subplots(1,1, figsize=(10,8))

for alg_no,alg in enumerate(alg_name):
    if alg_no<2:
        ax.plot(np.arange(1,11),btes[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3)
    else:
        ax.plot(np.arange(1,11),btes[alg_no], c=c[alg_no], label=alg_name[alg_no])

ax.set_yticks([.9,.95, 1, 1.05,1.1])
ax.set_ylim([0.87,1.12])
ax.set_xticks(np.arange(1,11))
ax.tick_params(labelsize=20)
ax.set_xlabel('Number of tasks seen', fontsize=24)
ax.set_ylabel('Transfer Efficiency', fontsize=24)
ax.set_title("Label Shuffled CIFAR", fontsize = 24)
ax.hlines(1,1,10, colors='grey', linestyles='dashed',linewidth=1.5)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
plt.tight_layout()
ax.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=22)
plt.savefig('result/figs/label_shufffle.pdf', dpi=500)
# %%
