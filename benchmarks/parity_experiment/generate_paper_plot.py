#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import seaborn as sns
import matplotlib
import numpy as np
import pickle
from sklearn.datasets import make_blobs
#from proglearn.sims import generate_gaussian_parity

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil
from matplotlib.ticker import ScalarFormatter

#%%
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c

def _generate_2d_rotation(theta=0):
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    return R

def generate_gaussian_parity(
    n_samples,
    centers=None,
    class_label=None,
    cluster_std=0.25,
    angle_params=None,
    random_state=None,
):
    """
    Generate 2-dimensional Gaussian XOR distribution.
    (Classic XOR problem but each point is the
    center of a Gaussian blob distribution)
    Parameters
    ----------
    n_samples : int
        Total number of points divided among the four
        clusters with equal probability.
    centers : array of shape [n_centers,2], optional (default=None)
        The coordinates of the ceneter of total n_centers blobs.
    class_label : array of shape [n_centers], optional (default=None)
        class label for each blob.
    cluster_std : float, optional (default=1)
        The standard deviation of the blobs.
    angle_params: float, optional (default=None)
        Number of radians to rotate the distribution by.
    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """

    if random_state != None:
        np.random.seed(random_state)

    if centers == None:
        centers = np.array([(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)])

    if class_label == None:
        class_label = [0, 1, 1, 0]

    blob_num = len(class_label)

    # get the number of samples in each blob with equal probability
    samples_per_blob = np.random.multinomial(
        n_samples, 1 / blob_num * np.ones(blob_num)
    )

    X, y = make_blobs(
        n_samples=samples_per_blob,
        n_features=2,
        centers=centers,
        cluster_std=cluster_std,
    )

    for blob in range(blob_num):
        y[np.where(y == blob)] = class_label[blob]

    if angle_params != None:
        R = _generate_2d_rotation(angle_params)
        X = X @ R

    return X, y

def move_avg(x, w):
    avg = []
    y = []
    i = 0
    for x_ in x:
        i +=1
        avg.append(x_)

        if i>w:
            avg.remove(avg[0])
        y.append(np.mean(avg))
    return y


#%%#%% Plotting the result
# mc_rep = 50
fontsize = 30
labelsize = 28


fig = plt.figure(constrained_layout=True, figsize=(25, 23))
gs = fig.add_gridspec(23, 25)

colors = sns.color_palette("Dark2", n_colors=2)

X, Y = generate_gaussian_parity(750)
Z, W = generate_gaussian_parity(750, angle_params=np.pi / 2)
P, Q = generate_gaussian_parity(750, angle_params=np.pi / 4)

ax = fig.add_subplot(gs[:6, 2:8])
ax.scatter(X[:, 0], X[:, 1], c=get_colors(colors, Y), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Ai. Gaussian XOR", fontsize=30)

plt.tight_layout()
ax.axis("off")
# plt.savefig('./result/figs/gaussian-xor.pdf')

#####################
ax = fig.add_subplot(gs[:6, 10:16])
ax.scatter(Z[:, 0], Z[:, 1], c=get_colors(colors, W), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Aii. Gaussian XNOR", fontsize=30)
ax.axis("off")

#####################
ax = fig.add_subplot(gs[:6, 18:24])
ax.scatter(P[:, 0], P[:, 1], c=get_colors(colors, Q), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Aiii. Gaussian R-XOR", fontsize=30)
ax.axis("off")

######################
mean_error = unpickle("data/mean_xor_nxor_with_rep.pickle")
mean_error_nn = unpickle("data/mean_xor_nxor_nn.pickle")

n_xor = (100 * np.arange(0.5, 7.5, step=0.25)).astype(int)
n_nxor = (100 * np.arange(0.25, 7.5, step=0.25)).astype(int)

n1s = n_xor
n2s = n_nxor

ns = np.concatenate((n1s, n2s + n1s[-1]))
ls = ["-", "--"]
algorithms = ["XOR Forest", "N-XOR Forest", "SynF", "RF", "SynN", "DN"]


TASK1 = "XOR"
TASK2 = "XNOR"

fontsize = 30
labelsize = 28

colors = sns.color_palette("Set1", n_colors=2)

ax1 = fig.add_subplot(gs[7:13, 2:8])

ax1.plot(
    ns,
    mean_error[1],
    label=algorithms[2],
    c=colors[0],
    ls=ls[np.sum(1 > 1).astype(int)],
    lw=3,
)
ax1.plot(
    ns,
    move_avg(mean_error_nn[1],4),
    label=algorithms[4],
    c='#377eb8',
    ls=ls[np.sum(1 > 1).astype(int)],
    lw=3,
)

ax1.plot(
    ns,
    mean_error[4],
    label=algorithms[3],
    c="g",
    ls=ls[np.sum(1 > 1).astype(int)],
    lw=3,
)
ax1.plot(
    ns,
    move_avg(mean_error_nn[4],4),
    label=algorithms[5],
    c="#b15928",
    ls=ls[np.sum(1 > 1).astype(int)],
    lw=3,
)

ax1.set_ylabel("Generalization Error (%s)" % (TASK1), fontsize=fontsize)
ax1.legend(loc="upper left", fontsize=20, frameon=False)
# ax1.set_ylim(0.09, 0.21)
ax1.set_xlabel("Total Sample Size", fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
ax1.set_xticks([50, 750, 1500])
# ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
ax1.set_title("Bi. XOR", fontsize=30)

right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)

ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(ScalarFormatter())
ax1.set_yticks([0.1, 0.3, 0.5])
ax1.text(400, np.mean(ax1.get_ylim()), "%s" % (TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s" % (TASK2), fontsize=26)

#######################
mean_error = unpickle("data/mean_xor_nxor_with_rep.pickle")
mean_error_nn = unpickle("data/mean_xor_nxor_nn.pickle")

algorithms = ["XOR Forest", "N-XOR Forest", "SynF", "RF", "SynN", "DN"]

TASK1 = "XOR"
TASK2 = "XNOR"

ax1 = fig.add_subplot(gs[7:13, 10:16])

ax1.plot(
    ns[len(n1s) :],
    mean_error[3, len(n1s) :],
    label=algorithms[2],
    c=colors[0],
    lw=3,
)
ax1.plot(
    ns[len(n1s) :],
    move_avg(mean_error_nn[3, len(n1s) :], 4),
    label=algorithms[4],
    c='#377eb8',
    lw=3,
)

ax1.plot(
    ns[len(n1s) :],
    mean_error[5, len(n1s) :],
    label=algorithms[3],
    c="g",
    lw=3,
)
ax1.plot(
    ns[len(n1s) :],
    move_avg(mean_error_nn[5, len(n1s) :], 4),
    label=algorithms[5],
    c="#b15928",
    lw=3,
)

ax1.set_ylabel("Generalization Error (%s)" % (TASK2), fontsize=fontsize)
# ax1.legend(loc='upper left', fontsize=18, frameon=False)
#         ax1.set_ylim(-0.01, 0.22)
ax1.set_xlabel("Total Sample Size", fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
# ax1.set_yticks([0.15, 0.2])
ax1.set_xticks([50, 750, 1500])
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")

# ax1.set_ylim(0.11, 0.21)

right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)

# ax1.set_ylim(0.14, 0.36)
ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(ScalarFormatter())
ax1.set_yticks([0.1, 0.5, 0.9])
ax1.text(400, np.mean(ax1.get_ylim()), "%s" % (TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s" % (TASK2), fontsize=26)

ax1.set_title("Bii. XNOR", fontsize=30)

##################
mean_te = unpickle("data/mean_te_xor_nxor_with_rep.pickle")
mean_te_nn = unpickle("data/mean_te_xor_nxor_nn.pickle")

algorithms = ["SynF BLE", "SynF FLE", "RF BLE", "RF FLE", "SynN BLE", "SynN FLE", "DN BLE", "DN FLE"]

TASK1 = "XOR"
TASK2 = "XNOR"

ax1 = fig.add_subplot(gs[7:13, 18:24])

ax1.plot(ns, mean_te[0], label=algorithms[0], c=colors[0], ls=ls[0], lw=3)
ax1.plot(ns, move_avg(mean_te_nn[0],4), label=algorithms[4], c='#377eb8', ls=ls[0], lw=3)
'''sns.regplot(
    ns,
    mean_te_nn[0],
    label=algorithms[4],
    color="#377eb8",
    line_kws={"linestyle":ls[0],
    "linewidth":3},
    ax=ax1,
    lowess=True,
    scatter=False
)'''


ax1.plot(
    ns[len(n1s) :],
    mean_te[1, len(n1s) :],
    label=algorithms[1],
    c=colors[0],
    ls=ls[1],
    lw=3,
)
ax1.plot(
    ns[len(n1s) :],
    move_avg(mean_te_nn[1, len(n1s) :],4),
    label=algorithms[5],
    c="#377eb8",
    ls=ls[1],
    lw=3,
)
'''sns.regplot(
    ns[len(n1s) :],
    mean_te_nn[1, len(n1s) :],
    label=algorithms[5],
    color="#377eb8",
    line_kws={"linestyle":ls[1],
    "linewidth":3},
    ax=ax1,
    lowess=True,
    scatter=False
)'''

ax1.plot(ns, mean_te[2], label=algorithms[2], c="g", ls=ls[0], lw=3)
ax1.plot(ns, move_avg(mean_te_nn[2],4), label=algorithms[6], c="#b15928", ls=ls[0], lw=3)
'''sns.regplot(
    ns,
    mean_te_nn[2],
    label=algorithms[6],
    color="#b15928",
    line_kws={"linestyle":ls[0],
    "linewidth":3},
    ax=ax1,
    lowess=True,
    scatter=False
)'''

ax1.plot(
    ns[len(n1s) :], mean_te[3, len(n1s) :], label=algorithms[3], c="g", ls=ls[1], lw=3
)
ax1.plot(
    ns[len(n1s) :], move_avg(mean_te_nn[3, len(n1s) :], 4), label=algorithms[7], c="#b15928", ls=ls[1], lw=3
)
'''sns.regplot(
    ns[len(n1s) :],
    mean_te_nn[3, len(n1s) :],
    label=algorithms[7],
    color="#b15928",
    line_kws={"linestyle":ls[1],
    "linewidth":3},
    ax=ax1,
    lowess=True,
    scatter=False
)'''

ax1.set_ylabel("Forward / Backward Learning", fontsize=fontsize)
ax1.legend(loc="upper left", fontsize=20, frameon=False)
ax1.set_yticks([0.05, 1, 2.5])
ax1.set_ylim(0.05, 2.52)
ax1.set_xlabel("Total Sample Size", fontsize=fontsize)
log_lbl = np.round(
    np.log([.05,1,2.5]),
    2
)
labels = [item.get_text() for item in ax1.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax1.set_yticklabels(labels)

ax1.tick_params(labelsize=labelsize)
ax1.set_xticks([50, 750, 1500])
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)
ax1.hlines(1, 50, 1500, colors="gray", linestyles="dashed", linewidth=1.5)

ax1.text(400, np.mean(ax1.get_ylim())-.7, "%s" % (TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim())-.7, "%s" % (TASK2), fontsize=26)
ax1.set_title("Biii.", fontsize=30)
######################
mean_te = unpickle("data/mean_te_xor_rxor_with_rep.pickle")
mean_te_nn = unpickle("data/mean_te_xor_rxor_nn.pickle")

algorithms = ["SynF BLE", "SynF FLE", "RF BLE", "RF FLE", "SynN BLE", "SynN FLE", "DN BLE", "DN FLE"]

TASK1 = "XOR"
TASK2 = "R-XOR"

ax1 = fig.add_subplot(gs[15:21, 2:8])

ax1.plot(ns, mean_te[0], label=algorithms[0], c=colors[0], ls=ls[0], lw=3)
ax1.plot(ns, move_avg(mean_te_nn[0],4), label=algorithms[4], c='#377eb8', ls=ls[0], lw=3)

ax1.plot(
    ns[len(n1s) :],
    mean_te[1, len(n1s) :],
    label=algorithms[1],
    c=colors[0],
    lw=3,
    ls=ls[1]
)
ax1.plot(
    ns[len(n1s) :],
    move_avg(mean_te_nn[1, len(n1s) :],4),
    label=algorithms[5],
    c='#377eb8',
    lw=3,
    ls=ls[1]
)

ax1.plot(ns, mean_te[2], label=algorithms[2], c="g", ls=ls[0], lw=3)
ax1.plot(ns, move_avg(mean_te_nn[2], 4), label=algorithms[6], c="#b15928", ls=ls[0], lw=3)
ax1.plot(
    ns[len(n1s) :], mean_te[3, len(n1s) :], label=algorithms[3], c="g", ls=ls[1], lw=3
)
ax1.plot(
    ns[len(n1s) :], move_avg(mean_te_nn[3, len(n1s) :], 4), label=algorithms[7], c="#b15928", ls=ls[1], lw=3
)

ax1.set_ylabel("Forward / Backward Learning", fontsize=fontsize)
# ax1.legend(loc='upper left', fontsize=20, frameon=False)
ax1.set_ylim(0.2, 1.2)
ax1.set_xlabel("Total Sample Size", fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
ax1.set_yticks([0.2, 0.6, 1, 1.2])
ax1.set_xticks([50, 750, 1500])
ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)
ax1.hlines(1, 50, 1500, colors="gray", linestyles="dashed", linewidth=1.5)

ax1.set_xlabel("Total Sample Size", fontsize=fontsize)
log_lbl = np.round(
    np.log([.2,.6,1,1.2]),
    2
)
labels = [item.get_text() for item in ax1.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax1.set_yticklabels(labels)

ax1.text(400, np.mean(ax1.get_ylim()), "%s" % (TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s" % (TASK2), fontsize=26)
ax1.set_title("Ci.", fontsize=30)
########################################################
ax = fig.add_subplot(gs[15:21, 10:16])
with open("data/mean_angle_te_with_rep.pickle", "rb") as f:
    te = pickle.load(f)

with open("data/mean_angle_te_nn.pickle", "rb") as f:
    te_nn = pickle.load(f)

angle_sweep = range(0, 90, 1)

ax.plot(angle_sweep, te, c="r", linewidth=3)
ax.plot(angle_sweep, move_avg(te_nn[:-1],10), c='#377eb8', linewidth=3)
ax.set_xticks(range(0, 91, 45))
ax.tick_params(labelsize=labelsize)
ax.set_xlabel("Angle of Rotation (Degrees)", fontsize=fontsize)
ax.set_ylabel("log BLE (XOR)", fontsize=fontsize)
ax.set_ylim(0.89, 1.25)
ax.set_yticks([0.9, 1, 1.1, 1.2])

log_lbl = np.round(
    np.log([0.9, 1, 1.1, 1.2]),
    2
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)

# ax.set_title("XOR vs. Rotated-XOR", fontsize = fontsize)
ax.hlines(1, 0, 90, colors="grey", linestyles="dashed", linewidth=1.5)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.set_title("Cii.", fontsize=30)
#####################################
ax = fig.add_subplot(gs[15:21, 18:24])

'''with open("data/mean_sample_te100.pickle", "rb") as f:
    te100 = pickle.load(f)
with open("data/mean_sample_te1000.pickle", "rb") as f:
    te1000 = pickle.load(f)'''
with open("data/mean_sample_te100.pickle", "rb") as f:
    te = pickle.load(f)
with open("data/mean_sample_te_nn.pickle", "rb") as f:
    te_nn = 1./pickle.load(f)


task2_sample_sweep = (2 ** np.arange(np.log2(60), np.log2(5010) + 1, 0.25)).astype(
    "int"
)

ax.plot(task2_sample_sweep, te, c="r", linewidth=3, label='SynF')
ax.plot(task2_sample_sweep, move_avg(te_nn,10), c='#377eb8', linewidth=3, label='SynF')

ax.hlines(1, 60, 5500, colors="gray", linestyles="dashed", linewidth=1.5)
ax.set_xscale("log")
#ax.set_xticks([])
ax.set_yticks([0.95, 1, 1.1, 1.25])

log_lbl = np.round(
    np.log([0.95, 1, 1.1, 1.25]),
    2
)
labels = [item.get_text() for item in ax.get_yticklabels()]

for ii,_ in enumerate(labels):
    labels[ii] = str(log_lbl[ii])

ax.set_yticklabels(labels)

ax.tick_params(labelsize=26)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
'''ax.text(50, np.mean(ax.get_ylim()) - 0.082, "50", fontsize=labelsize)
ax.text(500, np.mean(ax.get_ylim()) - 0.082, "500", fontsize=labelsize)
ax.text(5000, np.mean(ax.get_ylim()) - 0.082, "5000", fontsize=labelsize)
'''
ax.text(
    30,
    np.mean(ax.get_ylim()) - 0.22,
    "Number of $25^\circ$-RXOR Training Samples",
    fontsize=fontsize - 4,
)
ax.set_ylabel("log BLE (XOR)", fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
ax.set_title("Ciii.", fontsize=30)
#ax.legend(fontsize=fontsize-5, frameon=False)

plt.savefig("./plots/parity_exp.pdf")
# %%

# %%
