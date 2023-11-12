#%%
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from itertools import product
import pandas as pd

import sys
from sklearn.model_selection import StratifiedKFold
from math import log2, ceil

import joblib
from joblib import Parallel, delayed
from multiprocessing import Pool

from pympler.asizeof import asizeof

from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import (
    NeuralClassificationTransformer,
    TreeClassificationTransformer,
)
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter

import tensorflow as tf

import time
# %%
def get_size(obj, seen=None):
    print(obj)
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    #print(seen)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        print([get_size(v, seen) for v in obj.values()], 'size')
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
        print(get_size(obj.__dict__, seen), 'size')
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
        print([get_size(i, seen) for i in obj], 'seen')
    return size

#%%
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

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

# %%
def LF_experiment(
    train_x,
    train_y,
    ntrees,
    shift,
    slot,
    model,
    num_points_per_task,
    acorn=None,
):

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
            "euclidean_layer_idx": -1,
            "optimizer": keras.optimizers.Adam(3e-4),
        }

        default_voter_class = KNNClassificationVoter
        default_voter_kwargs = {"k": int(np.log2(num_points_per_task))}

        default_decider_class = SimpleArgmaxAverage
    elif model == "uf":
        default_transformer_class = TreeClassificationTransformer
        default_transformer_kwargs = {
            "kwargs": {"max_depth": 30, "max_features": "auto"}
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

        progressive_learner.add_task(
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
            num_transformers=1 if model == "dnn" else ntrees,
            transformer_voter_decider_split=[0.67, 0.33, 0],
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
        #joblib.dump(progressive_learner, 'model_'+str(task_ii)+'.pickle')
        print(get_size(progressive_learner)/(1024**2), get_size(progressive_learner.task_id_to_transformer_id_to_voters)/1024**2, get_size(progressive_learner.transformer_id_to_transformers)/1024**2)
        print(objsize.get_deep_size(progressive_learner)/(1024**2), objsize.get_deep_size(progressive_learner.task_id_to_transformer_id_to_voters)/1024**2, objsize.get_deep_size(progressive_learner.transformer_id_to_transformers)/1024**2)

    print(get_size(progressive_learner)/(1024**2))  

# %%
model = "dnn"

#%%
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
if model == "uf":
    data_x = data_x.reshape(
        (data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3])
    )
data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]

train_x, train_y, test_x, test_y = cross_val_data(
        data_x, data_y, 500, shift=1
    )
# %%
LF_experiment(
            train_x,
            train_y,
            10,
            1,
            1,
            model,
            500,
            acorn=12345,
        )
# %%
'''LF_experiment(
            train_x,
            train_y,
            10,
            1,
            1,
            'dnn',
            500,
            acorn=12345,
        )'''
# %%
