#%%
import random
import tensorflow as tf
import keras
from keras import layers
from itertools import product
import pandas as pd

import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil

from joblib import Parallel, delayed
from multiprocessing import Pool

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import NeuralClassificationTransformer, TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter

import tensorflow as tf

#%%
def LF_experiment(
    data_x, data_y, ntrees, shift, slot, model, num_points_per_task, acorn=None
):

    df = pd.DataFrame()
    shifts = []
    slots = []
    accuracies_across_tasks = []

    train_x_task0, train_y_task0, test_x_task0, test_y_task0 = cross_val_data(
        data_x, data_y, num_points_per_task, total_task=10, shift=shift, slot=slot
    )
    if model == "dnn":
        default_transformer_class = NeuralClassificationTransformer

        network = keras.Sequential()
        network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=np.shape(data_x)[1:]))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides = 2, padding = "same", activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides = 2, padding = "same", activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides = 2, padding = "same", activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=254, kernel_size=(3, 3), strides = 2, padding = "same", activation='relu'))

        network.add(layers.Flatten())
        network.add(layers.BatchNormalization())
        network.add(layers.Dense(2000, activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Dense(2000, activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Dense(units=10, activation = 'softmax'))

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
        default_voter_kwargs = {"k" : int(np.log2(500))}

        default_decider_class = SimpleArgmaxAverage

    elif model == "uf":
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
            num_transformers=1 if model == "dnn" else ntrees,
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

        print("Starting Task {} For Fold {} For Slot {}".format(task_ii, shift, slot))

        progressive_learner.add_task(
        X=train_x,
        y=train_y,
        num_transformers=1 if model == "dnn" else ntrees,
        transformer_voter_decider_split=[0.67, 0.33, 0],
        decider_kwargs={"classes": np.unique(train_y)}
        )

        task_0_predictions = progressive_learner.predict(test_x_task0, task_id=0)

        shifts.append(shift)
        slots.append(slot)
        accuracies_across_tasks.append(np.mean(task_0_predictions == test_y_task0))

        print("Accuracy Across Tasks: {}".format(accuracies_across_tasks))

    df["data_fold"] = shifts
    df["slot"] = slots
    df["accuracy"] = accuracies_across_tasks

    file_to_save = (
        "result/" + model + str(ntrees) + "_" + str(shift) + "_" + str(slot) + ".pickle"
    )
    with open(file_to_save, "wb") as f:
        pickle.dump(df, f)


#%%
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


#%%
### MAIN HYPERPARAMS ###
model = "dnn"
num_points_per_task = 500
########################

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
if model == "uf":
    data_x = data_x.reshape(
        (data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3])
    )
data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]

#%%
slot_fold = range(10)
shift_fold = range(1, 7, 1)

if model == "uf":
    n_trees = [10]
    iterable = product(n_trees, shift_fold, slot_fold)
    Parallel(n_jobs=-1, verbose=1)(
        delayed(LF_experiment)(
            data_x,
            data_y,
            ntree,
            shift,
            slot,
            model,
            num_points_per_task,
            acorn=12345
        )
        for ntree, shift, slot in iterable
    )
elif model == "dnn":
    iterable = product(shift_fold, slot_fold)

    for shift, slot in iterable:
        LF_experiment(
            data_x,
            data_y,
            0,
            shift,
            slot,
            model,
            num_points_per_task,
            acorn=12345
        )

# %%
