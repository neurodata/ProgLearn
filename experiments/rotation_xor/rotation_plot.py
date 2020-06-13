#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# %%
alg_name = ['L2F']
angles = np.arange(0,92,2)
tes = [[] for _ in range(len(alg_name))]

for algo_no,alg in enumerate(alg_name):
    for angle in angles:
        orig_error, transfer_error = pickle.load(
                open("results/angle_" + str(angle) + ".pickle", "rb")
                )
        tes[algo_no].append(orig_error / transfer_error)

with open('../plot_label_shuffled_angle_recruitment/rotation_result/res.pickle','wb') as f:
    pickle.dump(tes,f)

# %%
clr = ["#e41a1c"]
c = sns.color_palette(clr, n_colors=len(clr))
fig, ax = plt.subplots(1,1, figsize=(8,8))

for alg_no,alg in enumerate(alg_name):
    if alg_no<2:
        ax.plot(angles,tes[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3)
    else:
        ax.plot(angles,tes[alg_no], c=c[alg_no], label=alg_name[alg_no])


ax.set_xticks(range(0, 90 + 15, 15))
ax.tick_params(labelsize=20)
ax.set_xlabel('Angle of Rotation (Degrees)', fontsize=24)
ax.set_ylabel('Backward Transfer Efficiency', fontsize=24)
ax.set_title("XOR vs. Rotated-XOR", fontsize = 24)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
plt.tight_layout()
plt.hlines(1,1,90, colors='grey', linestyles='dashed',linewidth=1.5)
#x.legend(fontsize = 24)
plt.savefig('results/figs/rotation.pdf', dpi=500)
# %%
