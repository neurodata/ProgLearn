import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from proglearn.forest import LifelongClassificationForest
from sklearn.model_selection import train_test_split


def run_experiment(
    x_data, y_data, num_tasks, num_points_per_task, ntrees=10, model="uf", reps=100
):
    """Runs the FTE/BTE experiment.
    Referenced Chenyu's code, with modifications to adjust the number of tasks.
    """

    # initialize list for storing results
    accuracies_across_tasks = []

    # format data
    if model == "dnn":  # add dnn implementation in the future
        x = x_data
        y = y_data
    elif model == "uf":
        x = x_data.reshape(len(x_data), -1)
        y = y_data

    # get y values per task
    unique_y = np.unique(y_data)
    ys_by_task = unique_y.reshape(num_tasks, int(len(unique_y) / num_tasks))

    # run experiment over all reps
    for rep in range(reps):
        # print('Starting rep', rep)

        # for each task
        for task in range(num_tasks):

            # initialize progressive learner
            learner = LifelongClassificationForest(
                default_n_estimators=ntrees
            )  # default_max_depth=np.ceil(np.log2(num_points_per_task))

            # get train/test data (train = num_points_per_task)
            index = np.where(np.in1d(y, ys_by_task[task]))
            x_task0 = x[index]
            y_task0 = y[index]
            train_x_task0, test_x_task0, train_y_task0, test_y_task0 = train_test_split(
                x_task0, y_task0, test_size=0.25
            )
            train_x_task0 = train_x_task0[:num_points_per_task]
            train_y_task0 = train_y_task0[:num_points_per_task]

            # feed to learner and predict on single task
            learner.add_task(train_x_task0, train_y_task0)
            task_0_predictions = learner.predict(test_x_task0, task_id=0)
            accuracies_across_tasks.append(np.mean(task_0_predictions == test_y_task0))

            # evaluate for other tasks
            for other_task in range(num_tasks):

                if other_task == task:
                    pass

                else:

                    # get train/test data (train = num_points_per_task)
                    index = np.random.choice(
                        np.where(np.in1d(y, ys_by_task[other_task]))[0],
                        num_points_per_task,
                        replace=False,
                    )
                    train_x = x[index]
                    train_y = y[index]

                    # add transformer from other tasks
                    learner.add_task(train_x, train_y)

                # predict on current task using other tasks
                prev_task_predictions = learner.predict(test_x_task0, task_id=0)
                accuracies_across_tasks.append(
                    np.mean(prev_task_predictions == test_y_task0)
                )

    # average results
    accuracy_all_task = np.array(accuracies_across_tasks).reshape((reps, -1))
    accuracy_all_task = np.mean(accuracy_all_task, axis=0)

    return accuracy_all_task


def calculate_results(accuracy_all_task, num_tasks):
    """Calculates accuracy, BTE, FTE, and TE across each task.
    Referenced Chenyu's code, with modifications to adjust the number of tasks.
    """

    # accuracy
    acc = [[] for i in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            acc[i].append(accuracy_all_task[(num_tasks + 1) * i + (i + 1) + j])

    # backwards transfer efficiency
    bte = [[] for i in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            err_taskt_only = 1 - accuracy_all_task[(num_tasks + 1) * i]
            err_up_to_taskt = 1 - accuracy_all_task[(num_tasks + 1) * i + (i + 1)]
            err_all_seen = 1 - accuracy_all_task[(num_tasks + 1) * i + (i + 1) + j]
            bte[i].append(err_up_to_taskt / err_all_seen)

    # forwards transfer efficiency
    fte = [[] for i in range(num_tasks)]
    for i in range(num_tasks):
        err_taskt_only = 1 - accuracy_all_task[(num_tasks + 1) * i]
        err_up_to_taskt = 1 - accuracy_all_task[(num_tasks + 1) * i + (i + 1)]
        fte[i].append(err_taskt_only / err_up_to_taskt)

    # transfer efficiency
    te = [[] for i in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            err_taskt_only = 1 - accuracy_all_task[(num_tasks + 1) * i]
            err_all_seen = 1 - accuracy_all_task[(num_tasks + 1) * i + (i + 1) + j]
            te[i].append(err_taskt_only / err_all_seen)

    return acc, bte, fte, te


def plot_results(acc, bte, fte, te):
    """Plots the results of the FTE/BTE experiment."""

    # get number of tasks
    num_tasks = len(acc)

    # set figure parameters and plot results
    sns.set()
    clr = ["#e41a1c", "#a65628", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#CCCC00"]
    fontsize = 22
    ticksize = 20

    fig, ax = plt.subplots(2, 2, figsize=(16, 11.5))
    ax[0][0].plot(
        np.arange(1, num_tasks + 1),
        fte,
        c="red",
        marker=".",
        markersize=14,
        linewidth=3,
    )
    ax[0][0].hlines(1, 1, num_tasks, colors="grey", linestyles="dashed", linewidth=1.5)
    ax[0][0].tick_params(labelsize=ticksize)
    ax[0][0].set_xlabel("Number of tasks seen", fontsize=fontsize)
    ax[0][0].set_ylabel("FTE", fontsize=fontsize)

    for i in range(num_tasks):
        et = np.asarray(bte[i])
        ns = np.arange(i + 1, num_tasks + 1)
        ax[0][1].plot(ns, et, c="red", linewidth=2.6)

    ax[0][1].set_xlabel("Number of tasks seen", fontsize=fontsize)
    ax[0][1].set_ylabel("BTE", fontsize=fontsize)
    ax[0][1].tick_params(labelsize=ticksize)
    ax[0][1].hlines(1, 1, num_tasks, colors="grey", linestyles="dashed", linewidth=1.5)

    for i in range(num_tasks):
        et = np.asarray(te[i])
        ns = np.arange(i + 1, num_tasks + 1)
        ax[1][0].plot(ns, et, c="red", linewidth=2.6)

    ax[1][0].set_xlabel("Number of tasks seen", fontsize=fontsize)
    ax[1][0].set_ylabel("Transfer Efficiency", fontsize=fontsize)
    # ax[1][0].set_xticks(np.arange(1,10))
    ax[1][0].tick_params(labelsize=ticksize)
    ax[1][0].hlines(1, 1, num_tasks, colors="grey", linestyles="dashed", linewidth=1.5)

    for i in range(num_tasks):
        et = np.asarray(acc[i])
        ns = np.arange(i + 1, num_tasks + 1)
        ax[1][1].plot(ns, et, c="red", linewidth=2.6)

    ax[1][1].set_xlabel("Number of tasks seen", fontsize=fontsize)
    ax[1][1].set_ylabel("Accuracy", fontsize=fontsize)
    ax[1][1].tick_params(labelsize=ticksize)
