# Import the packages for experiment
import warnings

warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf

import time
import random
import keras

from keras import layers
from itertools import product
from math import log2, ceil
from joblib import Parallel, delayed
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold, train_test_split

# Import the progressive learning packages
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import (
    TreeClassificationTransformer,
    NeuralClassificationTransformer,
)
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter

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
    data_x, data_y, ntrees, shift, slot, model, num_points_per_task, acorn=None
):

    # construct dataframes
    df = pd.DataFrame()
    shifts = []
    slots = []
    accuracies_across_tasks = []
    train_times_across_tasks = []
    inference_times_across_tasks = []

    # randomly separate the training and testing subsets
    train_x_task0, train_y_task0, test_x_task0, test_y_task0 = cross_val_data(
        data_x, data_y, num_points_per_task, total_task=10, shift=shift, slot=slot
    )

    # choose Uncertainty Forest as transformer
    # construct the learner with model selected
    progressive_learner = ProgressiveLearner(
        default_transformer_class=TreeClassificationTransformer,
        default_transformer_kwargs={"kwargs": {"max_depth": 30}},
        default_voter_class=TreeClassificationVoter,
        default_voter_kwargs={},
        default_decider_class=SimpleArgmaxAverage,
    )

    # training process
    train_start_time = time.time()
    progressive_learner.add_task(
        X=train_x_task0,
        y=train_y_task0,
        num_transformers=ntrees,
        transformer_voter_decider_split=[0.67, 0.33, 0],
        decider_kwargs={"classes": np.unique(train_y_task0)},
    )
    train_end_time = time.time()

    # testing process
    inference_start_time = time.time()
    task_0_predictions = progressive_learner.predict(test_x_task0, task_id=0)
    inference_end_time = time.time()

    # record results
    shifts.append(shift)
    slots.append(slot)
    accuracies_across_tasks.append(np.mean(task_0_predictions == test_y_task0))
    train_times_across_tasks.append(train_end_time - train_start_time)
    inference_times_across_tasks.append(inference_end_time - inference_start_time)

    # repeating the tasks for 20 times
    for task_ii in range(1, 20):

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
        train_start_time = time.time()
        progressive_learner.add_transformer(
            X=train_x,
            y=train_y,
            transformer_data_proportion=1,
            num_transformers=ntrees,
            backward_task_ids=[0],
        )
        train_end_time = time.time()

        # testing process
        inference_start_time = time.time()
        task_0_predictions = progressive_learner.predict(test_x_task0, task_id=0)
        inference_end_time = time.time()

        # record results
        shifts.append(shift)
        slots.append(slot)
        accuracies_across_tasks.append(np.mean(task_0_predictions == test_y_task0))
        train_times_across_tasks.append(train_end_time - train_start_time)
        inference_times_across_tasks.append(inference_end_time - inference_start_time)

    # finalize dataframes
    df["data_fold"] = shifts
    df["slot"] = slots
    df["accuracy"] = accuracies_across_tasks
    df["train_times"] = train_times_across_tasks
    df["inference_times"] = inference_times_across_tasks

    # save results
    return df


# The method allows multiple lifelong forest experiments to be run at the same time
def run_parallel_exp(
    data_x, data_y, n_trees, model, num_points_per_task, slot=0, shift=1
):
    return L2_experiment(
        data_x,
        data_y,
        n_trees,
        shift,
        slot,
        model,
        num_points_per_task,
        acorn=12345,
    )


# The method calculates the bte and time results
def calculate_results(df_results, slot_num, shift_num):
    btes = []
    train_times = []
    inference_times = []

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
            train_times.append(df["train_times"])
            inference_times.append(df["inference_times"])

    return btes, train_times, inference_times


# Import the packages for plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})

# The method plots the Backward Transfer Efficiency
def plot_bte(bte):

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
    ax.legend(fontsize=22)

    # show the BTE plot
    plt.tight_layout()
    plt.show()


# The method plots the time changes
def plot_time(train_time_across_tasks, inference_time_across_tasks):

    # plot the recorded times
    ax = plt.subplot(111)

    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    ax.plot(
        range(len(train_time_across_tasks)),
        train_time_across_tasks,
        linewidth=3,
        linestyle="solid",
        label="Train Time",
    )
    ax.plot(
        range(len(inference_time_across_tasks)),
        inference_time_across_tasks,
        linewidth=3,
        linestyle="solid",
        label="Inference Time",
    )

    ax.set_xlabel("Number of Tasks Seen", fontsize=fontsize)
    ax.set_ylabel("Time (seconds)", fontsize=fontsize)
    ax.tick_params(labelsize=ticksize)
    ax.legend(fontsize=22)

    # show the time plot
    plt.tight_layout()
    plt.show()
