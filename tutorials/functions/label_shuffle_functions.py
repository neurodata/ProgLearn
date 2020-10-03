import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from proglearn.forest import LifelongClassificationForest


def run_parallel_exp(data_x, data_y, n_trees, num_points_per_task, slot=0, shift=1):

    train_x, train_y, test_x, test_y = cross_val_data(
        data_x, data_y, num_points_per_task, shift=shift
    )
    df = LF_experiment(
        train_x,
        train_y,
        test_x,
        test_y,
        n_trees,
        shift,
        slot,
        num_points_per_task,
        acorn=12345,
    )
    return df


def cross_val_data(data_x, data_y, num_points_per_task, total_task=10, shift=1):
    x = data_x.copy()
    y = data_y.copy()
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]

    batch_per_task = 5000 // num_points_per_task
    sample_per_class = num_points_per_task // total_task

    for task in range(total_task):
        for batch in range(batch_per_task):
            for class_no in range(task * 10, (task + 1) * 10, 1):
                indx = np.roll(idx[class_no], (shift - 1) * 100)

                if batch == 0 and class_no == 0 and task == 0:
                    train_x = x[
                        indx[batch * sample_per_class : (batch + 1) * sample_per_class]
                    ]
                    test_x = x[
                        indx[
                            batch * total_task
                            + num_points_per_task : (batch + 1) * total_task
                            + num_points_per_task
                        ]
                    ]
                    train_y = np.random.randint(
                        low=0, high=total_task, size=sample_per_class
                    )
                    test_y = np.random.randint(low=0, high=total_task, size=total_task)
                else:
                    train_x = np.concatenate(
                        (
                            train_x,
                            x[
                                indx[
                                    batch
                                    * sample_per_class : (batch + 1)
                                    * sample_per_class
                                ]
                            ],
                        ),
                        axis=0,
                    )
                    test_x = np.concatenate(
                        (
                            test_x,
                            x[
                                indx[
                                    batch * total_task
                                    + num_points_per_task : (batch + 1) * total_task
                                    + num_points_per_task
                                ]
                            ],
                        ),
                        axis=0,
                    )
                    if task == 0:
                        train_y = np.concatenate(
                            (
                                train_y,
                                y[
                                    indx[
                                        batch
                                        * sample_per_class : (batch + 1)
                                        * sample_per_class
                                    ]
                                ],
                            ),
                            axis=0,
                        )
                        test_y = np.concatenate(
                            (
                                test_y,
                                y[
                                    indx[
                                        batch * total_task
                                        + num_points_per_task : (batch + 1) * total_task
                                        + num_points_per_task
                                    ]
                                ],
                            ),
                            axis=0,
                        )
                    else:
                        train_y = np.concatenate(
                            (
                                train_y,
                                np.random.randint(
                                    low=0, high=total_task, size=sample_per_class
                                ),
                            ),
                            axis=0,
                        )
                        test_y = np.concatenate(
                            (
                                test_y,
                                np.random.randint(
                                    low=0, high=total_task, size=total_task
                                ),
                            ),
                            axis=0,
                        )

    return train_x, train_y, test_x, test_y


def LF_experiment(
    train_x,
    train_y,
    test_x,
    test_y,
    ntrees,
    shift,
    slot,
    num_points_per_task,
    acorn=None,
):

    # We initialize lists to store the results
    df = pd.DataFrame()
    shifts = []
    accuracies_across_tasks = []

    # Declare the progressive learner model (L2F), with ntrees as a parameter
    learner = LifelongClassificationForest(n_estimators=ntrees)

    for task_ii in range(10):
        print("Starting Task {} For Fold {} For Slot {}".format(task_ii, shift, slot))
        if acorn is not None:
            np.random.seed(acorn)

        # If task number is 0, add task. Else, add a transformer for the task
        if task_ii == 0:
            learner.add_task(
                X=train_x[
                    task_ii * 5000
                    + slot * num_points_per_task : task_ii * 5000
                    + (slot + 1) * num_points_per_task
                ],
                y=train_y[
                    task_ii * 5000
                    + slot * num_points_per_task : task_ii * 5000
                    + (slot + 1) * num_points_per_task
                ],
                task_id=0,
            )
        else:
            learner.add_transformer(
                X=train_x[
                    task_ii * 5000
                    + slot * num_points_per_task : task_ii * 5000
                    + (slot + 1) * num_points_per_task
                ],
                y=train_y[
                    task_ii * 5000
                    + slot * num_points_per_task : task_ii * 5000
                    + (slot + 1) * num_points_per_task
                ],
            )

        # Make a prediction on task 0 using the trained learner on test data
        llf_task = learner.predict(test_x[:1000], task_id=0)

        # Calculate the accuracy of the task 0 predictions
        acc = np.mean(llf_task == test_y[:1000])
        accuracies_across_tasks.append(acc)
        shifts.append(shift)

        print("Accuracy Across Tasks: {}".format(accuracies_across_tasks))

    df["data_fold"] = shifts
    df["task"] = range(1, 11)
    df["task_1_accuracy"] = accuracies_across_tasks

    return df


def get_bte(err):
    bte = []

    for i in range(10):
        bte.append(err[0] / err[i])

    return bte


def calc_bte(df_list, slots, shifts):
    shifts = shifts - 1
    reps = slots * shifts
    btes = np.zeros((1, 10), dtype=float)

    bte_tmp = [[] for _ in range(reps)]

    count = 0
    for shift in range(shifts):
        for slot in range(slots):

            # Get the dataframe containing the accuracies for the given shift and slot
            multitask_df = df_list[slot + shift * slots]
            err = []

            for ii in range(10):
                err.extend(
                    1
                    - np.array(
                        multitask_df[multitask_df["task"] == ii + 1]["task_1_accuracy"]
                    )
                )
            # Calculate the bte from task 1 error
            bte = get_bte(err)

            bte_tmp[count].extend(bte)
            count += 1

        # Calculate the mean backwards transfer efficiency
        btes[0] = np.mean(bte_tmp, axis=0)
    return btes


def plot_bte(btes):
    # Initialize the plot and color
    clr = ["#00008B"]
    c = sns.color_palette(clr, n_colors=len(clr))
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot the results
    ax.plot(np.arange(1, 11), btes[0], c=c[0], label="L2F", linewidth=3)

    # Format the plot, and show result
    ax.set_yticks([0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2])
    ax.set_xticks(np.arange(1, 11))
    ax.tick_params(labelsize=20)
    ax.set_xlabel("Number of tasks seen", fontsize=24)
    ax.set_ylabel("Backward Transfer Efficiency", fontsize=24)
    ax.set_title("Label Shuffled CIFAR", fontsize=24)
    ax.hlines(1, 1, 10, colors="grey", linestyles="dashed", linewidth=1.5)
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    plt.tight_layout()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=22)
    plt.show()
