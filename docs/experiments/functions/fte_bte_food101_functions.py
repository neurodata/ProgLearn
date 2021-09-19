import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from proglearn.forest import LifelongClassificationForest


def run_fte_bte_exp(data_x, data_y, which_task, ntrees=30, shift=0):

    df_total = []

    for slot in range(
        10
    ):  # Rotates the batch of training samples that are used from each class in each task
        train_x, train_y, test_x, test_y = cross_val_data(data_x, data_y, shift, slot)

        # Reshape the data
        train_x = train_x.reshape(
            train_x.shape[0], train_x.shape[1] * train_x.shape[2] * train_x.shape[3]
        )
        test_x = test_x.reshape(
            test_x.shape[0], test_x.shape[1] * test_x.shape[2] * test_x.shape[3]
        )

        df = fte_bte_experiment(
            train_x,
            train_y,
            test_x,
            test_y,
            ntrees,
            shift,
            slot,
            which_task,
            acorn=12345,
        )

        df_total.append(df)

    return df_total


def cross_val_data(data_x, data_y, shift, slot, total_cls=100):
    # Creates copies of both data_x and data_y so that they can be modified without affecting the original sets
    x = data_x.copy()
    y = data_y.copy()
    # Creates a sorted array of arrays that each contain the indices at which each unique element of data_y can be found
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]

    for i in range(total_cls):
        # Chooses the i'th array within the larger idx array
        indx = idx[i]
        ## The elements of indx are randomly shuffled
        # random.shuffle(indx)

        # 900 available training data points per class
        # Chooses all samples other than those in the testing batch
        tmp_x = np.concatenate(
            (x[indx[0 : (shift * 100)], :], x[indx[((shift + 1) * 100) : 1000], :]),
            axis=0,
        )
        tmp_y = np.concatenate(
            (y[indx[0 : (shift * 100)]], y[indx[((shift + 1) * 100) : 1000]]), axis=0
        )

        if i == 0:
            # 90 training data points per class
            # Rotates which set of 90 samples from each class is chosen for training each task
            # With 10 classes per task, total of 900 training samples per task
            train_x = tmp_x[(slot * 90) : ((slot + 1) * 90)]
            train_y = tmp_y[(slot * 90) : ((slot + 1) * 90)]

            # 100 testing data points per class
            # Batch for testing set is rotated each time
            test_x = x[indx[(shift * 100) : ((shift + 1) * 100)], :]
            test_y = y[indx[(shift * 100) : ((shift + 1) * 100)]]
        else:
            # 90 training data points per class
            # Rotates which set of 90 samples from each class is chosen for training each task
            # With 10 classes per task, total of 900 training samples per task
            train_x = np.concatenate(
                (train_x, tmp_x[(slot * 90) : ((slot + 1) * 90)]), axis=0
            )
            train_y = np.concatenate(
                (train_y, tmp_y[(slot * 90) : ((slot + 1) * 90)]), axis=0
            )

            # 100 testing data points per class
            # Batch for testing set is rotated each time
            test_x = np.concatenate(
                (test_x, x[indx[(shift * 100) : ((shift + 1) * 100)], :]), axis=0
            )
            test_y = np.concatenate(
                (test_y, y[indx[(shift * 100) : ((shift + 1) * 100)]]), axis=0
            )

    return train_x, train_y, test_x, test_y


def fte_bte_experiment(
    train_x,
    train_y,
    test_x,
    test_y,
    ntrees,
    shift,
    slot,
    which_task,
    acorn=None,
):

    # We initialize lists to store the results
    df = pd.DataFrame()
    accuracies_across_tasks = []

    # Declare the progressive learner model (L2F)
    learner = LifelongClassificationForest()

    for task_num in range((which_task - 1), 10):
        accuracy_per_task = []
        # print("Starting Task {} For Shift {} For Slot {}".format(task_num, shift, slot))
        if acorn is not None:
            np.random.seed(acorn)

        # If first task, add task. Else, add a transformer for the task
        if task_num == (which_task - 1):
            learner.add_task(
                X=train_x[(task_num * 900) : ((task_num + 1) * 900)],
                y=train_y[(task_num * 900) : ((task_num + 1) * 900)],
                task_id=0,
            )

            t_num = 0
            # Add transformers for all task up to current task (task t)
            while t_num < task_num:
                # Make a prediction on task t using the trained learner on test data
                llf_task = learner.predict(
                    test_x[((which_task - 1) * 1000) : (which_task * 1000), :],
                    task_id=0,
                )
                acc = np.mean(
                    llf_task == test_y[((which_task - 1) * 1000) : (which_task * 1000)]
                )
                accuracies_across_tasks.append(acc)

                learner.add_transformer(
                    X=train_x[(t_num * 900) : ((t_num + 1) * 900)],
                    y=train_y[(t_num * 900) : ((t_num + 1) * 900)],
                )

                # Add transformer for next task
                t_num = t_num + 1

        else:
            learner.add_transformer(
                X=train_x[(task_num * 900) : ((task_num + 1) * 900)],
                y=train_y[(task_num * 900) : ((task_num + 1) * 900)],
            )

        # Make a prediction on task t using the trained learner on test data
        llf_task = learner.predict(
            test_x[((which_task - 1) * 1000) : (which_task * 1000), :], task_id=0
        )
        acc = np.mean(
            llf_task == test_y[((which_task - 1) * 1000) : (which_task * 1000)]
        )
        accuracies_across_tasks.append(acc)
        # print("Accuracy Across Tasks: {}".format(accuracies_across_tasks))

    df["task"] = range(1, 11)
    df["task_accuracy"] = accuracies_across_tasks

    return df
