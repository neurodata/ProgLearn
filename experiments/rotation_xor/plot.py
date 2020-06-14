#%%
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
from itertools import product
import seaborn as sns
from matplotlib.pyplot import cm

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
        #print(single_err[i],err[i][i])
        fte.append(single_err[i]/err[i][i])
            
            
    return fte,bte,te     

fig, ax = plt.subplots(1,3, figsize=(32,10))

alg_name = ['L2F']
angles = np.arange(0,92,2)
tes = [[] for _ in range(len(alg_name))]

for algo_no,alg in enumerate(alg_name):
    for angle in angles:
        orig_error, transfer_error = pickle.load(
                open("bte_90/results/angle_" + str(angle) + ".pickle", "rb")
                )
        tes[algo_no].append(orig_error / transfer_error)

# %%
clr = ["#e41a1c"]
c = sns.color_palette(clr, n_colors=len(clr))

for alg_no,alg in enumerate(alg_name):
    if alg_no<2:
        ax[2].plot(angles,tes[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3)
    else:
        ax[2].plot(angles,tes[alg_no], c=c[alg_no], label=alg_name[alg_no])


ax[2].set_xticks(range(0, 90 + 15, 15))
ax[2].tick_params(labelsize=20)
ax[2].set_xlabel('Angle of Rotation (Degrees)', fontsize=24)
ax[2].set_ylabel('Backward Transfer Efficiency', fontsize=24)
ax[2].set_title("XOR vs. Rotated-XOR", fontsize = 24)
ax[2].hlines(1,0,90, colors='grey', linestyles='dashed',linewidth=1.5)


for num_task_1_data, linestyle in [(50, 'solid'), (100, 'dotted'), (200, 'dashed'), (500, 'dashdot')]:
    #%%
    te_across_reps = []
    for rep in range(10):
        filename = 'te_exp/result/'+str(num_task_1_data)+'_'+str(rep)+'.pickle'
        multitask_df, single_task_df = unpickle(filename)

        err = [[] for _ in range(10)]

        for ii in range(10):
            err[ii].extend(
             1 - np.array(
                 multitask_df[multitask_df['base_task']==ii+1]['accuracy']
             )
            )
        single_err = 1 - np.array(single_task_df['accuracy'])
        fte, bte, te = get_fte_bte(err,single_err)
        te_across_reps.append(te)

    #%%
    sns.set()

    n_tasks=10

    fontsize=22
    ticksize=20
    color=iter(cm.inferno(np.linspace(0,1,10)))
    
    for i in range(n_tasks):
        c = next(color)
        te_of_task = np.mean([te_across_reps[rep][i] for rep in range(10)], axis = 0)
        x_ticks = range(i + 1, n_tasks + 1)
        if i == 0:
            ax[1].plot(x_ticks, te_of_task, c=c, linewidth = 2.6, linestyle = linestyle, label = "n1 = " + str(num_task_1_data))
        else:
            ax[1].plot(x_ticks, te_of_task, c=c, linewidth = 2.6, linestyle = linestyle)

ax[1].legend(loc = "upper left", fontsize=15)
ax[1].hlines(1, 1,n_tasks, colors='grey', linestyles='dashed',linewidth=1.5)
ax[1].set_xlabel('Number of Tasks Seen', fontsize=24)
ax[1].set_ylabel('Transfer Efficiency', fontsize=24)
ax[1].set_title("Lifelong Forests on R-XOR Suite", fontsize = 24)

for a in ax:
    right_side = a.spines["right"]
    right_side.set_visible(False)
    top_side = a.spines["top"]
    top_side.set_visible(False)
plt.tight_layout()

plt.savefig('figs/rotation_te_exp.png')