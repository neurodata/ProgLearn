# Import the packages for experiment
import warnings

warnings.simplefilter("ignore")

import pandas as pd
import numpy as np

import random

from itertools import product
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

# Import the packages for plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})

# Import the progressive learning packages
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter

# The method randomly selects training and testing subsets from the original datasets,
# making cross-validation on program results possible.
def cross_val_data(
    data_x, data_y, num_points_per_task, total_task=10, shift=1, slot=0, task=0
):
    skf = StratifiedKFold(n_splits=6)
    for _ in range(shift + 1):
        train_idx, test_idx = next(skf.split(data_x, data_y))

    data_x_train, data_y_train = data_x[train_idx], data_y[train_idx]
    data_x_test, data_y_test = data_x[test_idx], data_y[test_idx]

    selected_classes = np.random.choice(range(0, 100), 10, replace=False)
    #print(selected_classes)
    train_idxs_of_selected_class = np.array(
        [np.where(data_y_train == y_val)[0] for y_val in selected_classes]
    )
    num_points_per_class_per_slot = [
        int(len(train_idxs_of_selected_class[class_idx]) // 10)
        for class_idx in range(len(selected_classes))
    ]
    selected_idxs = np.concatenate(
        [
            np.random.choice(
                train_idxs_of_selected_class[class_idx],
                num_points_per_class_per_slot[class_idx],
            )
            for class_idx in range(len(selected_classes))
        ]
    )
    train_idxs = np.random.choice(selected_idxs, num_points_per_task)
    data_x_train = data_x_train[train_idxs]
    data_y_train = data_y_train[train_idxs]

    test_idxs_of_selected_class = np.concatenate(
        [np.where(data_y_test == y_val)[0] for y_val in selected_classes]
    )
    data_x_test = data_x_test[test_idxs_of_selected_class]
    data_y_test = data_y_test[test_idxs_of_selected_class]
    #print(np.unique(data_y_train))
    return data_x_train, data_y_train, data_x_test, data_y_test


# The method runs the lifelong learning experiments
def Odif_experiment(
    data_x, data_y, ntrees, shift, slot, num_points_per_task, acorn=None
):

    df = pd.DataFrame()
    shifts = []
    slots = []
    accuracies_across_tasks = []

    train_x_task0, train_y_task0, test_x_task0, test_y_task0 = cross_val_data(
        data_x, data_y, num_points_per_task, total_task=10, shift=shift, slot=slot
    )

    default_transformer_class = TreeClassificationTransformer
    default_transformer_kwargs = {"kwargs" : {"max_depth" : 30, "max_features" : "auto"}}

    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = {}

    default_decider_class = SimpleArgmaxAverage


    progressive_learner = ProgressiveLearner(default_transformer_class = default_transformer_class,
                                     default_transformer_kwargs = default_transformer_kwargs,
                                     default_voter_class = default_voter_class,
                                     default_voter_kwargs = default_voter_kwargs,
                                     default_decider_class = default_decider_class)

    progressive_learner.add_task(
            X = train_x_task0,
            y = train_y_task0,
            num_transformers = ntrees,
            transformer_voter_decider_split = [0.67, 0.33, 0],
            decider_kwargs = {"classes" : np.unique(train_y_task0)}
        )


    task_0_predictions = progressive_learner.predict(test_x_task0, task_id=0)

    shifts.append(shift)
    slots.append(slot)
    accuracies_across_tasks.append(np.mean(task_0_predictions == test_y_task0))

    for task_ii in range(1, 20):
        train_x, train_y, _, _ = cross_val_data(
            data_x,
            data_y,
            num_points_per_task,
            total_task=10,
            shift=shift,
            slot=slot,
            task=task_ii
        )

        progressive_learner.add_task(
        X=train_x,
        y=train_y,
        num_transformers=ntrees,
        transformer_voter_decider_split=[0.67, 0.33, 0],
        decider_kwargs={"classes": np.unique(train_y)}
        )

        task_0_predictions = progressive_learner.predict(test_x_task0, task_id=0)

        shifts.append(shift)
        slots.append(slot)
        accuracies_across_tasks.append(np.mean(task_0_predictions == test_y_task0))

    df["data_fold"] = shifts
    df["slot"] = slots
    df["accuracy"] = accuracies_across_tasks

    return df 

# The method calculates the bte and time results
def calculate_results(df_results, slot_num, shift_num):
    btes = []

    # read the experiment results
    for slot in range(slot_num):
        for shift in range(shift_num):

            # load the dataframes
            df = df_results[shift * slot_num + slot]

            # calculate the backward transfer efficiency
            err = 1 - df["accuracy"]
            bte = err[0] / err

            # record information
            btes.append(bte)

    return btes

# The method plots the Backward Transfer Efficiency
def plot_bte(bte, fontsize, ticksize):

    # plot the BTE
    ax = plt.subplot(111)

    # hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    ax.plot(
        range(len(bte)),
        bte,
        c="red",
        linewidth=3,
        linestyle="solid",
        label="Odif",
    )
    ax.hlines(1, 0, 19, colors='gray', linestyles='dashed',linewidth=1.5)
    ax.set_xlabel("Number of Tasks Seen", fontsize=fontsize)
    ax.set_ylabel("log TE (Task 1)", fontsize=fontsize)
    ax.set_yticks([1, 1.1])
    ax.set_ylim(.98, 1.12)
    ax.set_xlabel("Number of Tasks Seen", fontsize=fontsize)
    log_lbl = np.round(
        np.log([1, 1.1]),
        2
    )
    labels = [item.get_text() for item in ax.get_yticklabels()]

    for ii,_ in enumerate(labels):
        labels[ii] = str(log_lbl[ii])

    ax.set_yticklabels(labels)

    ax.tick_params(labelsize=ticksize)
    ax.legend(fontsize=fontsize, frameon=False)

    # show the BTE plot
    plt.tight_layout()
    plt.show()
