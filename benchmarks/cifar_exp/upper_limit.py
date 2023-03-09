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

from joblib import Parallel, delayed
from multiprocessing import Pool

from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import (
    NeuralClassificationTransformer,
    TreeClassificationTransformer,
)
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter

import tensorflow as tf

import time
from sklearn.ensemble import RandomForestClassifier as rf 
# %%
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
res = []
n_trees = 100
num_sample_per_task = 500

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
data_x = data_x.reshape(
                (data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3])
            )
data_y = np.concatenate([y_train, y_test])

for shift in range(6):
    train_x, train_y, test_x, test_y = cross_val_data(
                data_x, data_y, num_sample_per_task, shift=shift
            )
    for slot in range(10):
        print("doing ", slot, shift)
        X, y  = train_x[slot * num_sample_per_task:(slot+1) * num_sample_per_task], train_y[slot * num_sample_per_task:(slot+1) * num_sample_per_task]
        X_test, y_test = test_x[:1000], test_y[:1000]

        for ii in range(1,10):
            X = np.concatenate((X, train_x[ii*5000+slot * num_sample_per_task:ii*5000+(slot+1) * num_sample_per_task]))
            y = np.concatenate((y, train_y[ii*5000+slot * num_sample_per_task:ii*5000+(slot+1) * num_sample_per_task]))
            X_test = np.concatenate((X_test, train_x[ii*1000:(ii+1)*1000]))
            y_test = np.concatenate((y_test, train_y[ii*1000:(ii+1)*1000]))

        model = rf(n_estimators=100)
        model.fit(X, y.ravel())

        res.append(
            np.mean(model.predict(X_test)==y_test.ravel())
        )
        print(res)
        break
    break

# %%
with open('RF_upper_limit.pickle', 'wb') as f:
    pickle.dump(res, f)
# %%
compile_kwargs = {
        "loss": "binary_crossentropy",
        "optimizer": keras.optimizers.Adam(3e-4),
    }
callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10, verbose=True)
fit_kwargs = {
        "epochs": 200,
        "batch_size": 32,
        "verbose": True,
        "callbacks": [callback],
    }

#%%
def getNN(input_size):
    network = keras.Sequential()
    network.add(
        layers.Conv2D(
            filters=16*10,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=input_size#np.shape(train_x)[1:],
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=32*10,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=64*10,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=128*10,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=254*10,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )

    network.add(layers.Flatten())
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(20000, activation="relu"))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(20000, activation="relu"))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(units=100, activation="softmax"))
    network.compile(**compile_kwargs)

    return network

#%%
res = []
num_sample_per_task = 500

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
data_y = np.concatenate([y_train, y_test])

for shift in range(6):
    train_x, train_y, test_x, test_y = cross_val_data(
                data_x, data_y, num_sample_per_task, shift=shift
            )
    for slot in range(10):
        print("doing ", slot, shift)
        X, y  = train_x[slot * num_sample_per_task:(slot+1) * num_sample_per_task], train_y[slot * num_sample_per_task:(slot+1) * num_sample_per_task]
        X_test, y_test = test_x[:1000], test_y[:1000]

        for ii in range(1,10):
            X = np.concatenate((X, train_x[ii*5000+slot * num_sample_per_task:ii*5000+(slot+1) * num_sample_per_task]))
            y = np.concatenate((y, train_y[ii*5000+slot * num_sample_per_task:ii*5000+(slot+1) * num_sample_per_task]))
            X_test = np.concatenate((X_test, train_x[ii*1000:(ii+1)*1000]))
            y_test = np.concatenate((y_test, train_y[ii*1000:(ii+1)*1000]))

        model = getNN(np.shape(X_test)[1:])
        model.fit(X, keras.utils.to_categorical(y.ravel()), **fit_kwargs)

        predicted_prob = model.predict(X_test)
        res.append(
            np.mean(np.argmax(predicted_prob, axis=1)==y_test.ravel())
        )
        print(res)
# %%
with open('DN_upper_limit.pickle', 'wb') as f:
    pickle.dump(f, res)
# %%
