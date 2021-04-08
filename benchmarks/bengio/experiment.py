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
from proglearn.transformers import NeuralClassificationTransformer, TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import tensorflow as tf

#%%
model = 'dnn'
folds = 5
# %%
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
data_x = np.concatenate([X_train, X_test])
data_x = data_x.astype("float32")/255
if model == "uf":
    data_x = data_x.reshape((data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3]))
data_y = np.concatenate([y_train, y_test])
data_x = np.expand_dims(data_x, -1)

task_A1 = (data_x[np.where((data_y==0) | (data_y==1) | (data_y==2))], data_y[np.where((data_y==0) | (data_y==1) | (data_y==2))])
task_A2 = (data_x[np.where((data_y==3) | (data_y==4) | (data_y==5))], data_y[np.where((data_y==3) | (data_y==4) | (data_y==5))])

(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
data_x = np.concatenate([X_train, X_test])
data_x = data_x.astype("float32")/255

if model == "uf":
    data_x = data_x.reshape((data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3]))
data_y = np.concatenate([y_train, y_test])
data_x = np.expand_dims(data_x, -1)

task_B = (data_x[np.where((data_y==6) | (data_y==7) | (data_y==8))], data_y[np.where((data_y==6) | (data_y==7) | (data_y==8))])
# %%
np.random.seed(12345)


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
network.add(layers.Dense(units=3, activation = 'softmax'))

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
default_voter_kwargs = {"k" : int(np.log2(17416))}

default_decider_class = SimpleArgmaxAverage



df = pd.DataFrame()
single_taskA1 = []
single_taskA2 = []
single_taskB = []
task_A1_representerA1 = []
task_A1_representerA2 = []
task_A1_representerB = []
llf_taskA1 = []
llf_taskA2 = []
llf_taskB = []
folding = []

skf = StratifiedKFold(n_splits=folds)

fold = 1
for (train_indxA1, test_indxA1), (train_indxA2, test_indxA2), (train_indxB, test_indxB) in zip(skf.split(task_A1[0],task_A1[1]), skf.split(task_A2[0],task_A2[1]), skf.split(task_B[0],task_B[1])):
    progressive_learner = ProgressiveLearner(default_transformer_class = default_transformer_class,
                                         default_transformer_kwargs = default_transformer_kwargs,
                                         default_voter_class = default_voter_class,
                                         default_voter_kwargs = default_voter_kwargs,
                                         default_decider_class = default_decider_class)
                                         
    progressive_learner.add_task(
            X = task_A1[0][train_indxA1],
            y = task_A1[1][train_indxA1],
            num_transformers = 1,
            transformer_voter_decider_split = [0.67, 0.33, 0],
            decider_kwargs = {"classes" : np.unique(task_A1[1][train_indxA1])}
            )

    progressive_learner.add_task(
        X = task_A2[0][train_indxA2],
        y = task_A2[1][train_indxA2],
        num_transformers = 1,
        transformer_voter_decider_split = [0.67, 0.33, 0],
        decider_kwargs = {"classes" : np.unique(task_A2[1][train_indxA2])}
        )

    progressive_learner.add_task(
        X = task_B[0][train_indxB],
        y = task_B[1][train_indxB],
        num_transformers = 1,
        transformer_voter_decider_split = [0.67, 0.33, 0],
        decider_kwargs = {"classes" : np.unique(task_B[1][train_indxB])}
        )

    predicted_label = progressive_learner.predict(task_A1[0][test_indxA1], transformer_ids=[0], task_id=0)
    single_taskA1.append(
        np.mean(predicted_label==task_A1[1][test_indxA1])
    )
    task_A1_representerA1.append(
        np.mean(predicted_label==task_A1[1][test_indxA1])
    )

    predicted_label = progressive_learner.predict(task_A2[0][test_indxA2], transformer_ids=[1], task_id=1)
    single_taskA2.append(
        np.mean(predicted_label==task_A2[1][test_indxA2])
    )

    predicted_label = progressive_learner.predict(task_B[0][test_indxB], transformer_ids=[2], task_id=2)
    single_taskB.append(
        np.mean(predicted_label==task_B[1][test_indxB])
    )

    predicted_label = progressive_learner.predict(task_A2[0][test_indxA2], transformer_ids=[0], task_id=1)
    task_A1_representerA2.append(
        np.mean(predicted_label==task_A2[1][test_indxA2])
    )

    predicted_label = progressive_learner.predict(task_B[0][test_indxB], transformer_ids=[0], task_id=2)
    task_A1_representerB.append(
        np.mean(predicted_label==task_B[1][test_indxB])
    )

    predicted_label = progressive_learner.predict(task_A1[0][test_indxA2], task_id=0)
    llf_taskA1.append(
        np.mean(predicted_label==task_A1[1][test_indxA1])
    )

    predicted_label = progressive_learner.predict(task_A2[0][test_indxA2], task_id=1)
    llf_taskA2.append(
        np.mean(predicted_label==task_A2[1][test_indxA2])
    )

    predicted_label = progressive_learner.predict(task_B[0][test_indxB], task_id=2)
    llf_taskB.append(
        np.mean(predicted_label==task_B[1][test_indxB])
    )

    folding.append(fold)
    fold += 1

df['single task A1'] = single_taskA1
df['single task A2'] = single_taskA2
df['single task B'] = single_taskB
df['representer A1 task A1'] = task_A1_representerA1
df['representer A1 task A2'] = task_A1_representerA2
df['representer A1 task B'] = task_A1_representerB
df['llf task A1'] = llf_taskA1
df['llf task A2'] = llf_taskA2
df['llf task B'] = llf_taskB
df['fold'] = folding

with open('./res.pickle','wb') as f:
    pickle.dump(df, f)
# %%
