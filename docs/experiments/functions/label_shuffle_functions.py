import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Import the progressive learning packages
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter


def run_parallel_exp(data_x, data_y, n_trees, num_points_per_task, slot=0, shift=1):

    train_x, train_y, test_x, test_y = cross_val_data(
        data_x, data_y, num_points_per_task, shift=shift
    )
    df = label_shuffle_experiment(
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
    test_data_slot = 100 // batch_per_task

    for task in range(total_task):
        for batch in range(batch_per_task):
            for class_no in range(task * 10, (task + 1) * 10, 1):
                indx = np.roll(idx[class_no], (shift - 1) * 100)

                if batch == 0 and class_no == 0 and task == 0:
                    train_x = x[
                        indx[batch * sample_per_class : (batch + 1) * sample_per_class],
                        :,
                    ]
                    train_y = y[
                        indx[batch * sample_per_class : (batch + 1) * sample_per_class]
                    ]
                    test_x = x[
                        indx[
                            batch * test_data_slot
                            + 500 : (batch + 1) * test_data_slot
                            + 500
                        ],
                        :,
                    ]
                    test_y = y[
                        indx[
                            batch * test_data_slot
                            + 500 : (batch + 1) * test_data_slot
                            + 500
                        ]
                    ]
                else:
                    train_x = np.concatenate(
                        (
                            train_x,
                            x[
                                indx[
                                    batch
                                    * sample_per_class : (batch + 1)
                                    * sample_per_class
                                ],
                                :,
                            ],
                        ),
                        axis=0,
                    )
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
                    test_x = np.concatenate(
                        (
                            test_x,
                            x[
                                indx[
                                    batch * test_data_slot
                                    + 500 : (batch + 1) * test_data_slot
                                    + 500
                                ],
                                :,
                            ],
                        ),
                        axis=0,
                    )
                    test_y = np.concatenate(
                        (
                            test_y,
                            y[
                                indx[
                                    batch * test_data_slot
                                    + 500 : (batch + 1) * test_data_slot
                                    + 500
                                ]
                            ],
                        ),
                        axis=0,
                    )

    return train_x, train_y, test_x, test_y


def label_shuffle_experiment(
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

    df = pd.DataFrame()
    single_task_accuracies = np.zeros(10, dtype=float)
    shifts = []
    tasks = []
    base_tasks = []
    accuracies_across_tasks = []

    default_transformer_class = TreeClassificationTransformer
    default_transformer_kwargs = {"kwargs": {"max_depth": 30, "max_features": "auto"}}

    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = {}

    default_decider_class = SimpleArgmaxAverage

    progressive_learner = ProgressiveLearner(
        default_transformer_class=default_transformer_class,
        default_transformer_kwargs=default_transformer_kwargs,
        default_voter_class=default_voter_class,
        default_voter_kwargs=default_voter_kwargs,
        default_decider_class=default_decider_class,
    )

    for task_ii in range(10):
        if acorn is not None:
            np.random.seed(acorn)

        tmp_y = train_y[
            task_ii * 5000
            + slot * num_points_per_task : task_ii * 5000
            + (slot + 1) * num_points_per_task
        ]

        if task_ii != 0:
            np.random.shuffle(tmp_y)

        progressive_learner.add_task(
            X=train_x[
                task_ii * 5000
                + slot * num_points_per_task : task_ii * 5000
                + (slot + 1) * num_points_per_task
            ],
            y=tmp_y,
            num_transformers=ntrees,
            transformer_voter_decider_split=[0.63, 0.37, 0],
            decider_kwargs={
                "classes": np.unique(
                    train_y[
                        task_ii * 5000
                        + slot * num_points_per_task : task_ii * 5000
                        + (slot + 1) * num_points_per_task
                    ]
                )
            },
        )

        llf_task = progressive_learner.predict(X=test_x[0:1000, :], task_id=0)

        shifts.append(shift)

        accuracies_across_tasks.append(np.mean(llf_task == test_y[0:1000]))

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
    ax.plot(np.arange(1, 11), btes[0], c=c[0], label="SynF", linewidth=3)

    # Format the plot, and show result
    ax.set_yticks([1, 1.1, 1.15])
    log_lbl = np.round(np.log([1, 1.1, 1.15]), 2)
    labels = [item.get_text() for item in ax.get_yticklabels()]

    for ii, _ in enumerate(labels):
        labels[ii] = str(log_lbl[ii])

    ax.set_yticklabels(labels)

    ax.set_xticks(np.arange(1, 11))
    ax.tick_params(labelsize=20)
    ax.set_xlabel("Number of tasks seen", fontsize=24)
    ax.set_ylabel("log Backward TE", fontsize=24)
    ax.set_title("Label Shuffled CIFAR", fontsize=24)
    ax.hlines(1, 1, 10, colors="grey", linestyles="dashed", linewidth=1.5)
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    plt.tight_layout()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=22)
    plt.show()
