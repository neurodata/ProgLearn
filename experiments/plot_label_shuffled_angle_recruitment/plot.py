#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# %%
fontsize=28
ticksize=32
fig, ax = plt.subplots(1,3, figsize=(58,20))

btes = unpickle('./label_shuffle_result/res.pickle')
alg_name = ['L2N','L2F','Prog_NN', 'DF_CNN','LwF','EWC','Online_EWC','SI']
clr = ["#00008B", "#e41a1c", "#a65628", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#CCCC00"]
c = sns.color_palette(clr, n_colors=len(clr))

for alg_no,alg in enumerate(alg_name):
    if alg_no<2:
        ax[0].plot(np.arange(1,11),btes[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3)
    else:
        ax[0].plot(np.arange(1,11),btes[alg_no], c=c[alg_no], label=alg_name[alg_no])

ax[0].set_yticks([.9,.95, 1, 1.05,1.1])
ax[0].set_ylim([0.87,1.12])
ax[0].set_xticks(np.arange(1,11))
ax[0].tick_params(labelsize=ticksize)
ax[0].set_xlabel('Number of tasks seen', fontsize=fontsize)
ax[0].set_ylabel('Transfer Efficiency', fontsize=fontsize)
ax[0].set_title("Label Shuffled CIFAR", fontsize = fontsize)
ax[0].hlines(1,1,10, colors='grey', linestyles='dashed',linewidth=1.5)
right_side = ax[0].spines["right"]
right_side.set_visible(False)
top_side = ax[0].spines["top"]
top_side.set_visible(False)
plt.tight_layout()


tes = unpickle('rotation_result/res.pickle')
angles = np.arange(0,180,2)
alg_name = ['L2N','L2F','LwF','EWC','Online_EWC','SI']
clr = ["#00008B", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#CCCC00"]
c = sns.color_palette(clr, n_colors=len(clr))

for alg_no,alg in enumerate(alg_name):
    if alg_no<2:
        ax[1].plot(angles,tes[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3)
    else:
        ax[1].plot(angles,tes[alg_no], c=c[alg_no], label=alg_name[alg_no])

ax[1].set_yticks([.9,.95, 1, 1.05])
ax[1].set_ylim([0.85,1.08])
ax[1].set_xticks([0,30,60,90,120,150,180])
ax[1].tick_params(labelsize=ticksize)
ax[1].set_xlabel('Angle of Rotation (Degrees)', fontsize=fontsize)
ax[1].set_ylabel('Backward Transfer Efficiency', fontsize=fontsize)
ax[1].set_title("Rotation Experiment", fontsize=fontsize)
right_side = ax[1].spines["right"]
right_side.set_visible(False)
top_side = ax[1].spines["top"]
top_side.set_visible(False)
plt.tight_layout()



mean_error = unpickle('recruitment_result/recruitment_mean.pickle')
std_error = unpickle('recruitment_result/recruitment_std.pickle')
ns = 10*np.array([50, 100, 200, 350, 500])
colors = sns.color_palette('Set1', n_colors=mean_error.shape[0]+2)

#labels = ['recruiting', 'Uncertainty Forest', 'hybrid', '50 Random', 'BF', 'building']
labels = ['hybrid', 'building', 'recruiting','50 Random', 'BF', 'Uncertainty Forest' ]
not_included = ['BF', '50 Random']
    
adjust = 0
for i, error_ in enumerate(mean_error[:-1]):
    if labels[i] in not_included:
        adjust +=1
        continue
    ax[2].plot(ns, mean_error[i], c=colors[i+1-adjust], label=labels[i])
    ax[2].fill_between(ns, 
            mean_error[i] + 1.96*std_error[i], 
            mean_error[i] - 1.96*std_error[i], 
            where=mean_error[i] + 1.96*std_error[i] >= mean_error[i] - 1.96*std_error[i], 
            facecolor=colors[i+1-adjust], 
            alpha=0.15,
            interpolate=False)

ax[2].plot(ns, mean_error[-1], c=colors[0], label=labels[-1])
ax[2].fill_between(ns, 
        mean_error[-1] + 1.96*std_error[-1], 
        mean_error[-1] - 1.96*std_error[-1], 
        where=mean_error[-1] + 1.96*std_error[i] >= mean_error[-1] - 1.96*std_error[-1], 
        facecolor=colors[0], 
        alpha=0.15,
        interpolate=False)


#ax.set_title('CIFAR Recruitment Experiment', fontsize=30)
ax[2].set_ylabel('Accuracy', fontsize=fontsize)
ax[2].set_xlabel('Number of Task 10 Samples', fontsize=fontsize)
ax[2].tick_params(labelsize=ticksize)
ax[2].set_ylim(0.325, 0.575)
ax[2].set_title("CIFAR Recruitment",fontsize=fontsize)
ax[2].set_xticks([500, 2000, 5000])
ax[2].set_yticks([0.35, 0.45, 0.55])

ax[2].legend(fontsize=24)

right_side = ax[2].spines["right"]
right_side.set_visible(False)
top_side = ax[2].spines["top"]
top_side.set_visible(False)

plt.savefig('figs/real_adversary_recruit.pdf', dpi=500)#

# %%
