import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.transformers import (
    NeuralClassificationTransformer,
    TreeClassificationTransformer,
)
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter
from sklearn.model_selection import train_test_split

from proglearn.forest import LifelongClassificationForest
from proglearn.network import LifelongClassificationNetwork

from tensorflow.keras.backend import clear_session  # To avoid OOM error when using dnn


def run_fte_bte_exp(data_x, data_y, which_task, model, ntrees=30, shift=0):

    df_total = []

    for slot in range(
        1
    ):  # Rotates the batch of training samples that are used from each class in each task
        train_x, train_y, test_x, test_y = cross_val_data(data_x, data_y, shift, slot)

        if model == "odif":
            # Reshape the data
            train_x = train_x.reshape(
                train_x.shape[0], train_x.shape[1] * train_x.shape[2] * train_x.shape[3]
            )
            test_x = test_x.reshape(
                test_x.shape[0], test_x.shape[1] * test_x.shape[2] * test_x.shape[3]
            )

        if model == "odin":
            clear_session()  # clear GPU memory before each run, to avoid OOM error

            default_transformer_class = NeuralClassificationTransformer

            network = keras.Sequential()
            network.add(
                layers.Conv2D(
                    filters=16,
                    kernel_size=(3, 3),
                    activation="relu",
                    input_shape=np.shape(data_x)[1:],
                )
            )
            network.add(layers.BatchNormalization())
            network.add(
                layers.Conv2D(
                    filters=32,
                    kernel_size=(3, 3),
                    strides=2,
                    padding="same",
                    activation="relu",
                )
            )
            network.add(layers.BatchNormalization())
            network.add(
                layers.Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    strides=2,
                    padding="same",
                    activation="relu",
                )
            )
            network.add(layers.BatchNormalization())
            network.add(
                layers.Conv2D(
                    filters=128,
                    kernel_size=(3, 3),
                    strides=2,
                    padding="same",
                    activation="relu",
                )
            )
            network.add(layers.BatchNormalization())
            network.add(
                layers.Conv2D(
                    filters=254,
                    kernel_size=(3, 3),
                    strides=2,
                    padding="same",
                    activation="relu",
                )
            )

            network.add(layers.Flatten())
            network.add(layers.BatchNormalization())
            network.add(layers.Dense(2000, activation="relu"))
            network.add(layers.BatchNormalization())
            network.add(layers.Dense(2000, activation="relu"))
            network.add(layers.BatchNormalization())
            network.add(layers.Dense(units=4, activation="softmax"))

            default_transformer_kwargs = {
                "network": network,
                "euclidean_layer_idx": -2,
                "loss": "categorical_crossentropy",
                "optimizer": Adam(3e-4),
                "fit_kwargs": {
                    "epochs": 100,
                    "callbacks": [EarlyStopping(patience=5, monitor="val_loss")],
                    "verbose": False,
                    "validation_split": 0.33,
                    "batch_size": 32,
                },
            }
            default_voter_class = KNNClassificationVoter
            default_voter_kwargs = {"k": int(np.log2(360))}
            default_decider_class = SimpleArgmaxAverage

            p_learner = ProgressiveLearner(
                default_transformer_class=default_transformer_class,
                default_transformer_kwargs=default_transformer_kwargs,
                default_voter_class=default_voter_class,
                default_voter_kwargs=default_voter_kwargs,
                default_decider_class=default_decider_class,
            )
        elif model == "odif":
            p_learner = LifelongClassificationForest()

        df = fte_bte_experiment(
            train_x,
            train_y,
            test_x,
            test_y,
            ntrees,
            shift,
            slot,
            model,
            p_learner,
            which_task,
            acorn=12345,
        )

        df_total.append(df)

    return df_total


def cross_val_data(data_x, data_y, shift, slot, total_cls=40):
    # Creates copies of both data_x and data_y so that they can be modified without affecting the original sets
    x = data_x.copy()
    y = data_y.copy()
    # Creates a sorted array of arrays that each contain the indices at which each unique element of data_y can be found
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]

    for i in range(total_cls):
        # Chooses the i'th array within the larger idx array
        indx = idx[i]

        # 90 available training data points per class
        # Chooses all samples other than those in the testing batch
        tmp_x = np.concatenate(
            (x[indx[0 : (shift * 30)], :], x[indx[((shift + 1) * 30) : 120], :]),
            axis=0,
        )
        tmp_y = np.concatenate(
            (y[indx[0 : (shift * 30)]], y[indx[((shift + 1) * 30) : 120]]), axis=0
        )

        if i == 0:
            # 90 training data points per class
            # Rotates which set of 90 samples from each class is chosen for training each task
            # With 4 classes per task, total of 360 training samples per task
            train_x = tmp_x[(slot * 90) : ((slot + 1) * 90)]
            train_y = tmp_y[(slot * 90) : ((slot + 1) * 90)]

            # 30 testing data points per class
            # Batch for testing set is rotated each time
            test_x = x[indx[(shift * 30) : ((shift + 1) * 30)], :]
            test_y = y[indx[(shift * 30) : ((shift + 1) * 30)]]
        else:
            # 90 training data points per class
            # Rotates which set of 90 samples from each class is chosen for training each task
            # With 4 classes per task, total of 360 training samples per task
            train_x = np.concatenate(
                (train_x, tmp_x[(slot * 90) : ((slot + 1) * 90)]), axis=0
            )
            train_y = np.concatenate(
                (train_y, tmp_y[(slot * 90) : ((slot + 1) * 90)]), axis=0
            )

            # 30 testing data points per class
            # Batch for testing set is rotated each time
            test_x = np.concatenate(
                (test_x, x[indx[(shift * 30) : ((shift + 1) * 30)], :]), axis=0
            )
            test_y = np.concatenate(
                (test_y, y[indx[(shift * 30) : ((shift + 1) * 30)]]), axis=0
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
    model,
    p_learner,
    which_task,
    acorn=None,
):

    # We initialize lists to store the results
    df = pd.DataFrame()
    accuracies_across_tasks = []

    # Declare the progressive learner model (Odif or Odin)
    learner = p_learner

    for task_num in range((which_task - 1), 10):
        accuracy_per_task = []
        # print("Starting Task {} For Shift {} For Slot {}".format(task_num, shift, slot))
        if acorn is not None:
            np.random.seed(acorn)

        # If first task, add task.
        if task_num == (which_task - 1):
            if model == "odin":
                learner.add_task(
                    X=train_x[(task_num * 360) : ((task_num + 1) * 360)],
                    y=train_y[(task_num * 360) : ((task_num + 1) * 360)],
                    task_id=task_num,
                    num_transformers=1,
                    transformer_voter_decider_split=[0.67, 0.33, 0],
                    decider_kwargs={
                        "classes": np.unique(
                            train_y[(task_num * 360) : ((task_num + 1) * 360)]
                        )
                    },
                )
            elif model == "odif":
                learner.add_task(
                    X=train_x[(task_num * 360) : ((task_num + 1) * 360)],
                    y=train_y[(task_num * 360) : ((task_num + 1) * 360)],
                    task_id=task_num,
                )

            t_num = 0
            # Add tasks for all tasks up to current task (task t)
            while t_num < task_num:
                # Make a prediction on task t using the trained learner on test data
                llf_task = learner.predict(
                    test_x[((which_task - 1) * 120) : (which_task * 120), :],
                    task_id=task_num,
                )
                acc = np.mean(
                    llf_task == test_y[((which_task - 1) * 120) : (which_task * 120)]
                )
                accuracies_across_tasks.append(acc)

                if model == "odin":
                    learner.add_task(
                        X=train_x[(task_num * 360) : ((task_num + 1) * 360)],
                        y=train_y[(task_num * 360) : ((task_num + 1) * 360)],
                        task_id=t_num,
                        num_transformers=1,
                        transformer_voter_decider_split=[0.67, 0.33, 0],
                        decider_kwargs={
                            "classes": np.unique(
                                train_y[(task_num * 360) : ((task_num + 1) * 360)]
                            )
                        },
                    )
                elif model == "odif":
                    learner.add_task(
                        X=train_x[(task_num * 360) : ((task_num + 1) * 360)],
                        y=train_y[(task_num * 360) : ((task_num + 1) * 360)],
                        task_id=t_num,
                    )

                # Add task for next task
                t_num = t_num + 1

        else:
            if model == "odin":
                learner.add_task(
                    X=train_x[(task_num * 360) : ((task_num + 1) * 360)],
                    y=train_y[(task_num * 360) : ((task_num + 1) * 360)],
                    task_id=task_num,
                    num_transformers=1,
                    transformer_voter_decider_split=[0.67, 0.33, 0],
                    decider_kwargs={
                        "classes": np.unique(
                            train_y[(task_num * 360) : ((task_num + 1) * 360)]
                        )
                    },
                )
            elif model == "odif":
                learner.add_task(
                    X=train_x[(task_num * 360) : ((task_num + 1) * 360)],
                    y=train_y[(task_num * 360) : ((task_num + 1) * 360)],
                    task_id=task_num,
                )

        # Make a prediction on task t using the trained learner on test data
        predictions = learner.predict(
            test_x[((which_task - 1) * 120) : (which_task * 120), :],
            task_id=(which_task - 1),
        )
        acc = np.mean(
            predictions == test_y[((which_task - 1) * 120) : (which_task * 120)]
        )
        accuracies_across_tasks.append(acc)
        # print("Accuracy Across Tasks: {}".format(accuracies_across_tasks))

    df["task"] = range(1, 11)
    df["task_accuracy"] = accuracies_across_tasks

    return df
