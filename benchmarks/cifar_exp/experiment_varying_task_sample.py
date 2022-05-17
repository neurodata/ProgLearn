#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from itertools import product
import pandas as pd

import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil

import sys
from joblib import Parallel, delayed
from multiprocessing import Pool

from proglearn import ProgressiveLearner
from proglearn import SimpleArgmaxAverage
from proglearn import NeuralClassificationTransformer, TreeClassificationTransformer
from proglearn import TreeClassificationVoter, KNNClassificationVoter

import tensorflow as tf
from numbers import Number
from collections import Set, Mapping, deque

import time

#%%
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


try:  # Python 2
    zero_depth_bases = (basestring, Number, xrange, bytearray)
    iteritems = "iteritems"
except NameError:  # Python 3
    zero_depth_bases = (str, bytes, Number, range, bytearray)
    iteritems = "items"


def getsize(obj_0):
    """Recursively iterate to sum size of object & members."""
    _seen_ids = set()

    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass  # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, "__dict__"):
            size += inner(vars(obj))
        if hasattr(obj, "__slots__"):  # can have __slots__ with __dict__
            size += sum(
                inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s)
            )
        return size

    return inner(obj_0)


#%%
def LF_experiment(
    train_x,
    train_y,
    test_x,
    test_y,
    ntrees,
    shift,
    slot,
    model,
    num_points_per_task,
    acorn=None,
):

    df = pd.DataFrame()
    single_task_accuracies = np.zeros(10, dtype=float)
    shifts = []
    tasks = []
    base_tasks = []
    accuracies_across_tasks = []
    train_times_across_tasks = []
    single_task_inference_times_across_tasks = []
    multitask_inference_times_across_tasks = []
    model_size = []

    if model == "dnn":
        default_transformer_class = NeuralClassificationTransformer

        network = keras.Sequential()
        network.add(
            layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=np.shape(train_x)[1:],
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
        network.add(layers.Dense(units=10, activation="softmax"))

        default_transformer_kwargs = {
            "network": network,
            "euclidean_layer_idx": -2,
            "num_classes": 10,
            "optimizer": keras.optimizers.Adam(3e-4),
        }

        default_voter_class = KNNClassificationVoter
        default_voter_kwargs = {"k": int(np.log2(num_points_per_task))}

        default_decider_class = SimpleArgmaxAverage
    elif model == "uf":
        default_transformer_class = TreeClassificationTransformer
        default_transformer_kwargs = {
            "kwargs": {"max_depth": None, "max_features": "auto"}
        }

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
        print("Starting Task {} For Fold {}".format(task_ii, shift))
        if acorn is not None:
            np.random.seed(acorn)

            reduced_sample_no = int(num_points_per_task * (1.29 ** task_ii))

            print(reduced_sample_no)

        train_start_time = time.time()
        progressive_learner.add_task(
            X=train_x[
                task_ii * 5000
                + slot * reduced_sample_no : task_ii * 5000
                + (slot + 1) * reduced_sample_no
            ],
            y=train_y[
                task_ii * 5000
                + slot * reduced_sample_no : task_ii * 5000
                + (slot + 1) * reduced_sample_no
            ],
            num_transformers=1 if model == "dnn" else ntrees,
            transformer_voter_decider_split=[0.67, 0.33, 0],
            decider_kwargs={
                "classes": np.unique(
                    train_y[
                        task_ii * 5000
                        + slot * reduced_sample_no : task_ii * 5000
                        + (slot + 1) * reduced_sample_no
                    ]
                )
            },
        )
        train_end_time = time.time()

        train_times_across_tasks.append(train_end_time - train_start_time)
        model_size.append(getsize(progressive_learner))
        print(model_size)

        single_task_inference_start_time = time.time()
        llf_task = progressive_learner.predict(
            X=test_x[task_ii * 1000 : (task_ii + 1) * 1000, :],
            transformer_ids=[task_ii],
            task_id=task_ii,
        )
        single_task_inference_end_time = time.time()
        single_task_accuracies[task_ii] = np.mean(
            llf_task == test_y[task_ii * 1000 : (task_ii + 1) * 1000]
        )
        single_task_inference_times_across_tasks.append(
            single_task_inference_end_time - single_task_inference_start_time
        )

        for task_jj in range(task_ii + 1):
            multitask_inference_start_time = time.time()
            llf_task = progressive_learner.predict(
                X=test_x[task_jj * 1000 : (task_jj + 1) * 1000, :], task_id=task_jj
            )
            multitask_inference_end_time = time.time()

            shifts.append(shift)
            tasks.append(task_jj + 1)
            base_tasks.append(task_ii + 1)
            accuracies_across_tasks.append(
                np.mean(llf_task == test_y[task_jj * 1000 : (task_jj + 1) * 1000])
            )
            multitask_inference_times_across_tasks.append(
                multitask_inference_end_time - multitask_inference_start_time
            )

    df["data_fold"] = shifts
    df["task"] = tasks
    df["base_task"] = base_tasks
    df["accuracy"] = accuracies_across_tasks
    df["multitask_inference_times"] = multitask_inference_times_across_tasks

    df_single_task = pd.DataFrame()
    df_single_task["task"] = range(1, 11)
    df_single_task["data_fold"] = shift
    df_single_task["accuracy"] = single_task_accuracies
    df_single_task[
        "single_task_inference_times"
    ] = single_task_inference_times_across_tasks
    df_single_task["train_times"] = train_times_across_tasks
    df_single_task["model_size"] = model_size

    summary = (df, df_single_task)
    file_to_save = (
        "result/result/increased_sample_"
        + model
        + str(ntrees)
        + "_"
        + str(shift)
        + "_"
        + str(slot)
        + ".pickle"
    )
    with open(file_to_save, "wb") as f:
        pickle.dump(summary, f)


#%%
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


#%%
def run_parallel_exp(
    data_x, data_y, n_trees, model, num_points_per_task, slot=0, shift=1
):
    train_x, train_y, test_x, test_y = cross_val_data(data_x, data_y, 500, shift=shift)

    if model == "dnn":
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        with tf.device("/gpu:" + str(shift % 4)):
            LF_experiment(
                train_x,
                train_y,
                test_x,
                test_y,
                n_trees,
                shift,
                slot,
                model,
                num_points_per_task,
                acorn=12345,
            )
    else:
        LF_experiment(
            train_x,
            train_y,
            test_x,
            test_y,
            n_trees,
            shift,
            slot,
            model,
            num_points_per_task,
            acorn=12345,
        )


#%%
### MAIN HYPERPARAMS ###
model = "dnn"
num_points_per_task = 500
total_slots = 1  # 5000//num_points_per_task
###########################

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
if model == "uf":
    data_x = data_x.reshape(
        (data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3])
    )
data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]


#%%
if model == "uf":
    slot_fold = range(total_slots)
    shift_fold = range(1, 7, 1)
    n_trees = [10]
    iterable = product(n_trees, shift_fold, slot_fold)
    Parallel(n_jobs=-2, verbose=1)(
        delayed(run_parallel_exp)(
            data_x, data_y, ntree, model, num_points_per_task, slot=slot, shift=shift
        )
        for ntree, shift, slot in iterable
    )
elif model == "dnn":
    slot_fold = range(total_slots)

    def perform_shift(shift_slot_tuple):
        shift, slot = shift_slot_tuple
        return run_parallel_exp(
            data_x, data_y, 0, model, num_points_per_task, slot=slot, shift=shift
        )

    print("Performing Stage 1 Shifts")
    stage_1_shifts = range(1, 5)
    stage_1_iterable = product(stage_1_shifts, slot_fold)
    with Pool(4) as p:
        p.map(perform_shift, stage_1_iterable)

    print("Performing Stage 2 Shifts")
    stage_2_shifts = range(5, 7)
    stage_2_iterable = product(stage_2_shifts, slot_fold)
    with Pool(4) as p:
        p.map(perform_shift, stage_2_iterable)
