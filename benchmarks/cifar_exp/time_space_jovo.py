#%%
import random
import tensorflow as tf
import keras
from keras import layers
from itertools import product
import pandas as pd

import numpy as np
import pickle

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

import time
import sys
# %%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_size(obj, seen=None):
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
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    '''elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])'''
    return size

def cross_val_data(data_x, data_y, num_points_per_task, total_task=10, shift=1):
    x = data_x.copy()
    y = data_y.copy()
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]

    batch_per_task=5000//num_points_per_task
    sample_per_class = num_points_per_task//total_task
    test_data_slot=100//batch_per_task

    for task in range(total_task):
        for batch in range(batch_per_task):
            for class_no in range(task*10,(task+1)*10,1):
                indx = np.roll(idx[class_no],(shift-1)*100)

                if batch==0 and class_no==0 and task==0:
                    train_x = x[indx[batch*sample_per_class:(batch+1)*sample_per_class],:]
                    train_y = y[indx[batch*sample_per_class:(batch+1)*sample_per_class]]
                    test_x = x[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500],:]
                    test_y = y[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500]]
                else:
                    train_x = np.concatenate((train_x, x[indx[batch*sample_per_class:(batch+1)*sample_per_class],:]), axis=0)
                    train_y = np.concatenate((train_y, y[indx[batch*sample_per_class:(batch+1)*sample_per_class]]), axis=0)
                    test_x = np.concatenate((test_x, x[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500],:]), axis=0)
                    test_y = np.concatenate((test_y, y[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500]]), axis=0)

    return train_x, train_y, test_x, test_y
# %%
if __name__ == "__main__":
    ## main hyperparameters ##
    model = "uf"
    ntrees = 10
    num_points_per_task = 500
    mem_info = []
    time_info = []
    task_sample = range(5000,0,-500)
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
    data_x = np.concatenate([X_train, X_test])

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
    data_x = np.concatenate([X_train, X_test])
    if model == "uf":
        data_x = data_x.reshape((data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3]))
    data_y = np.concatenate([y_train, y_test])
    data_y = data_y[:, 0]

    train_x, train_y, test_x, test_y = cross_val_data(data_x, data_y, 500, shift=1)

    if model == "dnn":
        default_transformer_class = NeuralClassificationTransformer

        network = keras.Sequential()
        network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=np.shape(X_train)[1:]))
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
        default_voter_kwargs = {"k" : int(np.log2(num_points_per_task))}

        default_decider_class = SimpleArgmaxAverage
    elif model == "uf":
        default_transformer_class = TreeClassificationTransformer
        default_transformer_kwargs = {"kwargs" : {"max_depth" : None, "max_features" : "auto"}}

        default_voter_class = TreeClassificationVoter
        default_voter_kwargs = {}

        default_decider_class = SimpleArgmaxAverage


    progressive_learner = ProgressiveLearner(default_transformer_class = default_transformer_class,
                                         default_transformer_kwargs = default_transformer_kwargs,
                                         default_voter_class = default_voter_class,
                                         default_voter_kwargs = default_voter_kwargs,
                                         default_decider_class = default_decider_class)

# %%
train_start_time = time.time()
for n_sample in task_sample:
    progressive_learner.add_task(
            X = train_x[0:n_sample],
            y = train_y[0:n_sample],
            num_transformers = 1 if model == "dnn" else ntrees,
            transformer_voter_decider_split = [0.63, 0.37, 0],
            decider_kwargs = {"classes" : [np.unique(train_y[0:n_sample])]}
            )

    train_end_time = time.time()

    time_info.append(train_end_time-train_start_time)
    mem_info.append(get_size(progressive_learner))

with open('./result/mem_res/'+model+'_same_machine.pickle','wb') as f:
    pickle.dump(mem_info,f)

with open('./result/time_res/'+model+'_same_machine.pickle','wb') as f:
    pickle.dump(time_info,f)
# %%
