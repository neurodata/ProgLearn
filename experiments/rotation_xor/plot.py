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

fig, ax = plt.subplots(1,2, figsize=(18, 8))

alg_name = ['L2F']
angles = np.arange(0,91,1)
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
        ax[0].plot(angles,tes[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3)
    else:
        ax[0].plot(angles,tes[alg_no], c=c[alg_no], label=alg_name[alg_no])


ax[0].set_xticks(range(0, 90 + 15, 15))
ax[0].tick_params(labelsize=25)
ax[0].set_xlabel('Angle of Rotation (Degrees)', fontsize=24)
ax[0].set_ylabel('Backward Transfer Efficiency', fontsize=24)
ax[0].set_title("XOR vs. Rotated-XOR", fontsize = 24)
ax[0].hlines(1,0,90, colors='grey', linestyles='dashed',linewidth=1.5)



#%%
te_ra = []
n1_ra = range(10, 5000, 50)
for n1 in n1_ra:
    te_across_reps = []
    for rep in range(500):
        filename = 'te_exp/result/'+str(n1)+'_'+str(rep)+'.pickle'
        df = unpickle(filename)
        te_across_reps.append(float(df['te']))
    te_ra.append(np.mean(te_across_reps))


#%%
sns.set()

fontsize=22
ticksize=20

ax[1].plot(n1_ra, te_ra, c="#e41a1c", linewidth = 2.6)
ax[1].tick_params(labelsize=25)
ax[1].hlines(1, 1, max(n1_ra), colors='grey', linestyles='dashed',linewidth=1.5)
ax[1].set_xlabel('Number of Task 1 Training Samples', fontsize=24)
ax[1].set_ylabel('Backward Transfer Efficiency', fontsize=24)
ax[1].set_title("Training Set Size Effect", fontsize = 24)

for a in ax:
    right_side = a.spines["right"]
    right_side.set_visible(False)
    top_side = a.spines["top"]
    top_side.set_visible(False)
plt.tight_layout()

plt.savefig('figs/rotation_te_exp.pdf')