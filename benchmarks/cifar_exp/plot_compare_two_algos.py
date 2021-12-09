#%%
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})
import numpy as np
from itertools import product
import seaborn as sns

### MAIN HYPERPARAMS ###
ntrees = 0
shifts = 6
task_num = 10
model = "dnn"
########################
algo1_name = "SupervisedContrastiveLoss"
algo2_name = "CategoricalCrossEntropy"
#%%
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def get_fte_bte(err, single_err, ntrees):
    bte = [[] for i in range(10)]
    te = [[] for i in range(10)]
    fte = []

    for i in range(10):
        for j in range(i, 10):
            # print(err[j][i],j,i)
            bte[i].append(err[i][i] / err[j][i])
            te[i].append(single_err[i] / err[j][i])

    for i in range(10):
        # print(single_err[i],err[i][i])
        fte.append(single_err[i] / err[i][i])

    return fte, bte, te


def calc_mean_bte(btes, task_num=10, reps=6):
    mean_bte = [[] for i in range(task_num)]

    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(btes[i][j])

        tmp = tmp / reps
        mean_bte[j].extend(tmp)

    return mean_bte


def calc_mean_te(tes, task_num=10, reps=6):
    mean_te = [[] for i in range(task_num)]

    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(tes[i][j])

        tmp = tmp / reps
        mean_te[j].extend(tmp)

    return mean_te


def calc_mean_fte(ftes, task_num=10, reps=6):
    fte = np.asarray(ftes)

    return list(np.mean(np.asarray(fte), axis=0))


def calc_mean_err(err, task_num=10, reps=6):
    mean_err = [[] for i in range(task_num)]

    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(err[i][j])

        tmp = tmp / reps
        # print(tmp)
        mean_err[j].extend([tmp])

    return mean_err


def calc_mean_multitask_time(multitask_time, task_num=10, reps=6):
    mean_multitask_time = [[] for i in range(task_num)]

    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(multitask_time[i][j])

        tmp = tmp / reps
        # print(tmp)
        mean_multitask_time[j].extend([tmp])

    return mean_multitask_time


#%%
reps = shifts

btes = [[] for i in range(task_num)]
ftes = [[] for i in range(task_num)]
tes = [[] for i in range(task_num)]
err_ = [[] for i in range(task_num)]
btes2 = [[] for i in range(task_num)]
ftes2 = [[] for i in range(task_num)]
tes2 = [[] for i in range(task_num)]
err_2 = [[] for i in range(task_num)]


te_tmp = [[] for _ in range(reps)]
bte_tmp = [[] for _ in range(reps)]
fte_tmp = [[] for _ in range(reps)]
err_tmp = [[] for _ in range(reps)]
train_time_tmp = [[] for _ in range(reps)]
single_task_inference_time_tmp = [[] for _ in range(reps)]
multitask_inference_time_tmp = [[] for _ in range(reps)]
te_tmp2 = [[] for _ in range(reps)]
bte_tmp2 = [[] for _ in range(reps)]
fte_tmp2 = [[] for _ in range(reps)]
err_tmp2 = [[] for _ in range(reps)]
train_time_tmp2 = [[] for _ in range(reps)]
single_task_inference_time_tmp2 = [[] for _ in range(reps)]
multitask_inference_time_tmp2 = [[] for _ in range(reps)]

count = 0
for shift in range(shifts):
    filename = (
        "result/result/"
        + model
        + str(ntrees)
        + "_"
        + str(shift + 1)
        + "_"
        + algo1_name
        + ".pickle"
    )
    filename2 = (
        "result/result/"
        + model
        + str(ntrees)
        + "_"
        + str(shift + 1)
        + "_"
        + algo2_name
        + ".pickle"
    )
    multitask_df, single_task_df = unpickle(filename)
    multitask_df2, single_task_df2 = unpickle(filename2)
    err = [[] for _ in range(10)]
    multitask_inference_times = [[] for _ in range(10)]
    err2 = [[] for _ in range(10)]
    multitask_inference_times2 = [[] for _ in range(10)]
    for ii in range(10):
        err[ii].extend(
            1 - np.array(multitask_df[multitask_df["base_task"] == ii + 1]["accuracy"])
        )
        err2[ii].extend(
            1
            - np.array(multitask_df2[multitask_df2["base_task"] == ii + 1]["accuracy"])
        )
        multitask_inference_times[ii].extend(
            np.array(
                multitask_df[multitask_df["base_task"] == ii + 1][
                    "multitask_inference_times"
                ]
            )
        )
        multitask_inference_times2[ii].extend(
            np.array(
                multitask_df2[multitask_df2["base_task"] == ii + 1][
                    "multitask_inference_times"
                ]
            )
        )
    single_err = 1 - np.array(single_task_df["accuracy"])
    single_err2 = 1 - np.array(single_task_df2["accuracy"])
    fte, bte, te = get_fte_bte(err, single_err, ntrees)
    fte2, bte2, te2 = get_fte_bte(err2, single_err2, ntrees)

    err_ = [[] for i in range(task_num)]
    for i in range(task_num):
        for j in range(task_num - i):
            # print(err[i+j][i])
            err_[i].append(err[i + j][i])
    err_2 = [[] for i in range(task_num)]
    for i in range(task_num):
        for j in range(task_num - i):
            # print(err[i+j][i])
            err_2[i].append(err2[i + j][i])

    train_time_tmp[count].extend(np.array(single_task_df["train_times"]))
    single_task_inference_time_tmp[count].extend(
        np.array(single_task_df["single_task_inference_times"])
    )
    multitask_inference_time_tmp[count].extend(multitask_inference_times)
    te_tmp[count].extend(te)
    bte_tmp[count].extend(bte)
    fte_tmp[count].extend(fte)
    err_tmp[count].extend(err_)
    train_time_tmp2[count].extend(np.array(single_task_df2["train_times"]))
    single_task_inference_time_tmp2[count].extend(
        np.array(single_task_df2["single_task_inference_times"])
    )
    multitask_inference_time_tmp2[count].extend(multitask_inference_times2)
    te_tmp2[count].extend(te2)
    bte_tmp2[count].extend(bte2)
    fte_tmp2[count].extend(fte2)
    err_tmp2[count].extend(err_2)
    count += 1

te = calc_mean_te(te_tmp, reps=reps)
bte = calc_mean_bte(bte_tmp, reps=reps)
fte = calc_mean_fte(fte_tmp, reps=reps)
error = calc_mean_err(err_tmp, reps=reps)
te2 = calc_mean_te(te_tmp2, reps=reps)
bte2 = calc_mean_bte(bte_tmp2, reps=reps)
fte2 = calc_mean_fte(fte_tmp2, reps=reps)
error2 = calc_mean_err(err_tmp2, reps=reps)

train_time = np.mean(train_time_tmp, axis=0)
single_task_inference_time = np.mean(single_task_inference_time_tmp, axis=0)
multitask_inference_time = calc_mean_multitask_time(multitask_inference_time_tmp)
multitask_inference_time = [
    np.mean(multitask_inference_time[i]) for i in range(len(multitask_inference_time))
]
train_time2 = np.mean(train_time_tmp2, axis=0)
single_task_inference_time2 = np.mean(single_task_inference_time_tmp2, axis=0)
multitask_inference_time2 = calc_mean_multitask_time(multitask_inference_time_tmp2)
multitask_inference_time2 = [
    np.mean(multitask_inference_time2[i]) for i in range(len(multitask_inference_time2))
]

#%%
sns.set()

n_tasks = 10
clr = ["#e41a1c", "#a65628", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#CCCC00"]
# c = sns.color_palette(clr, n_colors=len(clr))

fontsize = 22
ticksize = 20

fig, ax = plt.subplots(2, 2, figsize=(16, 11.5))
fig.suptitle(algo1_name + " - " + algo2_name, fontsize=25)
difference = []
zip_object = zip(fte2, fte)
for fte2_i, fte_i in zip_object:
    difference.append(fte2_i - fte_i)
ax[0][0].plot(
    np.arange(1, n_tasks + 1),
    difference,
    c="red",
    marker=".",
    markersize=14,
    linewidth=3,
)
ax[0][0].hlines(1, 1, n_tasks, colors="grey", linestyles="dashed", linewidth=1.5)
ax[0][0].tick_params(labelsize=ticksize)
ax[0][0].set_xlabel("Number of tasks seen", fontsize=fontsize)
ax[0][0].set_ylabel("FTE Difference", fontsize=fontsize)


for i in range(n_tasks):

    et = np.asarray(bte[i])
    et2 = np.asarray(bte2[i])
    ns = np.arange(i + 1, n_tasks + 1)
    ax[0][1].plot(ns, et2 - et, c="red", linewidth=2.6)

ax[0][1].set_xlabel("Number of tasks seen", fontsize=fontsize)
ax[0][1].set_ylabel("BTE Difference", fontsize=fontsize)
# ax[0][1].set_xticks(np.arange(1,10))
ax[0][1].tick_params(labelsize=ticksize)
ax[0][1].hlines(1, 1, n_tasks, colors="grey", linestyles="dashed", linewidth=1.5)


for i in range(n_tasks):

    et = np.asarray(te[i])
    et2 = np.asarray(te2[i])
    ns = np.arange(i + 1, n_tasks + 1)
    ax[1][0].plot(ns, et2 - et, c="red", linewidth=2.6)

ax[1][0].set_xlabel("Number of tasks seen", fontsize=fontsize)
ax[1][0].set_ylabel("TE Difference", fontsize=fontsize)
# ax[1][0].set_xticks(np.arange(1,10))
ax[1][0].tick_params(labelsize=ticksize)
ax[1][0].hlines(1, 1, n_tasks, colors="grey", linestyles="dashed", linewidth=1.5)


for i in range(n_tasks):
    et = np.asarray(error[i][0])
    et2 = np.asarray(error2[i][0])
    ns = np.arange(i + 1, n_tasks + 1)

    ax[1][1].plot(ns, 1 - et2 - (1 - et), c="red", linewidth=2.6)

# ax[1][1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=22)
ax[1][1].set_xlabel("Number of tasks seen", fontsize=fontsize)
ax[1][1].set_ylabel("Accuracy Difference", fontsize=fontsize)
# ax[1][1].set_yticks([.4,.6,.8,.9,1, 1.1,1.2])
# ax[1][1].set_xticks(np.arange(1,10))
# ax[1][1].set_ylim(0.89, 1.15)
ax[1][1].tick_params(labelsize=ticksize)

plt.savefig("result/result/", dpi=300)
plt.close()

ax = plt.subplot(111)

# Hide the right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

# fig.suptitle('ntrees = '+str(ntrees),fontsize=25)
ax.plot(
    range(len(train_time)),
    train_time,
    linewidth=3,
    linestyle="solid",
    label="Train Time",
)
ax.plot(
    range(len(single_task_inference_time)),
    single_task_inference_time,
    linewidth=3,
    linestyle="solid",
    label="Single Task Inference Time",
)
ax.plot(
    range(len(multitask_inference_time)),
    multitask_inference_time,
    linewidth=3,
    linestyle="solid",
    label="Multi-Task Inference Time",
)


ax.set_xlabel("Number of Tasks Seen", fontsize=fontsize)
ax.set_ylabel("Time (seconds)", fontsize=fontsize)
ax.tick_params(labelsize=ticksize)
ax.legend(fontsize=22)

plt.tight_layout()
# plt.savefig('result/result/',dpi=300)

# %%
