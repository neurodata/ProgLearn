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

    selected_classes = np.random.choice(range(0, 100), total_task)
    train_idxs_of_selected_class = np.array(
        [np.where(data_y_train == y_val)[0] for y_val in selected_classes]
    )
    num_points_per_class_per_slot = [
        int(len(train_idxs_of_selected_class[class_idx]) // total_task)
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

    return data_x_train, data_y_train, data_x_test, data_y_test


# The method runs the lifelong learning experiments
def L2_experiment(
    data_x, data_y, ntrees, shift, slot, num_points_per_task, task_num, acorn=None
):

    # construct dataframes
    df = pd.DataFrame()
    shifts = []
    slots = []
    accuracies_across_tasks = []

    # randomly separate the training and testing subsets
    train_x_task0, train_y_task0, test_x_task0, test_y_task0 = cross_val_data(
        data_x, data_y, num_points_per_task, total_task=10, shift=shift, slot=slot
    )

    # choose Uncertainty Forest as transformer
    progressive_learner = ProgressiveLearner(
        default_transformer_class=TreeClassificationTransformer,
        default_transformer_kwargs={"kwargs": {"max_depth": 30}},
        default_voter_class=TreeClassificationVoter,
        default_voter_kwargs={},
        default_decider_class=SimpleArgmaxAverage,
    )

    # training process
    progressive_learner.add_task(
        X=train_x_task0,
        y=train_y_task0,
        num_transformers=ntrees,
        transformer_voter_decider_split=[0.67, 0.33, 0],
        decider_kwargs={"classes": np.unique(train_y_task0)},
    )

    # testing process
    task_0_predictions = progressive_learner.predict(test_x_task0, task_id=0)

    # record results
    shifts.append(shift)
    slots.append(slot)
    accuracies_across_tasks.append(np.mean(task_0_predictions == test_y_task0))

    # repeating the tasks for task_num times
    for task_ii in range(1, task_num):

        # randomly separate the training and testing subsets
        train_x, train_y, _, _ = cross_val_data(
            data_x,
            data_y,
            num_points_per_task,
            total_task=10,
            shift=shift,
            slot=slot,
            task=task_ii,
        )

        # training process
        progressive_learner.add_transformer(
            X=train_x,
            y=train_y,
            transformer_data_proportion=1,
            num_transformers=ntrees,
            backward_task_ids=[0],
        )

        # testing process
        task_0_predictions = progressive_learner.predict(test_x_task0, task_id=0)

        # record results
        shifts.append(shift)
        slots.append(slot)
        accuracies_across_tasks.append(np.mean(task_0_predictions == test_y_task0))

    # finalize dataframes
    df["data_fold"] = shifts
    df["slot"] = slots
    df["accuracy"] = accuracies_across_tasks

    # save results
    return df


# The method allows multiple lifelong forest experiments to be run at the same time
def run_parallel_exp(
    data_x, data_y, n_trees, num_points_per_task, task_num, slot=0, shift=1
):
    return L2_experiment(
        data_x,
        data_y,
        n_trees,
        shift,
        slot,
        num_points_per_task,
        task_num,
        acorn=12345,
    )


# The method calculates the bte and time results
def calculate_results(df_results, slot_num, shift_num):
    first = []
    total = []

    # read the experiment results
    for slot in range(slot_num):
        for shift in range(shift_num):

            # load the dataframes
            df = df_results[shift * slot_num + slot]

            # calculate the backward transfer efficiency
            err = 1 - df["accuracy"]

            # record information
            first.append(err[0])
            total.append(err)

    # average the errors
    ave_first = np.mean(first, axis=0)
    ave_total = np.mean(total, axis=0)
    btes = np.divide(ave_first, ave_total)

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
        label="L2F",
    )

    ax.set_xlabel("Number of Tasks Seen", fontsize=fontsize)
    ax.set_ylabel("BTE (Task 1)", fontsize=fontsize)
    ax.set_yticks([1.0, 1.1, 1.2])
    ax.tick_params(labelsize=ticksize)
    ax.legend(fontsize=fontsize)

    # show the BTE plot
    plt.tight_layout()
    plt.show()
