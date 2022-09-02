#%%
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pickle
import keras

# Import the progressive learning packages
from keras import layers
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.transformers import (
    NeuralClassificationTransformer,
    TreeClassificationTransformer,
)
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter
from sklearn.model_selection import train_test_split
from keras.backend import clear_session 

# %%
def experiment(model='synf', ntrees=10):
    num_tasks = 5
    tasks = []
    base_tasks = []
    accuracies_across_tasks = []
    singletask_accuracy = []
    df_multitask = pd.DataFrame()
    df_singletask = pd.DataFrame()
    #these are actually five_datasets paths, name changed to fool DFCNN
    train_path = '/Users/jayantadey/DF-CNN/Data/cifar-100-python/train.pickle'
    test_path = '/Users/jayantadey/DF-CNN/Data/cifar-100-python/test.pickle'

    with open(train_path, 'rb') as f:
        df_train = pickle.load(f)

    with open(test_path, 'rb') as f:
        df_test = pickle.load(f)

    train_x, train_y, test_x, test_y = df_train[b'data'], np.array(df_train[b'fine_labels']),\
        df_test[b'data'], np.array(df_test[b'fine_labels'])

    if model == "synn":

        clear_session()  # clear GPU memory before each run, to avoid OOM error

        train_x = train_x.reshape(-1, 3, 32, 32)
        test_x = test_x.reshape(-1, 3, 32, 32)


        default_transformer_class = NeuralClassificationTransformer

        network = keras.Sequential()
        network.add(
            layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=np.shape(train_x[0]),
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
        network.add(layers.Dense(units=10, activation="softmax"))  # units=10

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
        default_voter_kwargs = {"k": int(np.log2(50000))}
        default_decider_class = SimpleArgmaxAverage

    elif model == "synf":
        default_transformer_class = TreeClassificationTransformer
        default_transformer_kwargs = {"kwargs": {"max_depth": 30}}
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

    for task in range(num_tasks):
        train_idx = np.where(np.logical_and(train_y>=task*10, train_y<(task+1)*10))[0]
        test_idx = np.where(np.logical_and(test_y>=task*10, test_y<(task+1)*10))[0]

        print("doing task ", task)
        progressive_learner.add_task(
            X=train_x[train_idx],
            y=train_y[train_idx],
            task_id=task,
            num_transformers=1 if model == "synn" else ntrees,
            transformer_voter_decider_split=[0.67, 0.33, 0],
            decider_kwargs={"classes": np.unique(train_y[train_idx])},
        )

        singletask_prediction = progressive_learner.predict(
            X=test_x[test_idx], transformer_ids=[task], task_id=task
        )
        singletask_accuracy.append(
            np.mean(singletask_prediction==test_y[test_idx])
        )
        print('accuracy ',np.mean(singletask_prediction==test_y[test_idx]))
        for ii in range(task+1):
            test_idx = np.where(np.logical_and(test_y>=ii*10, test_y<(ii+1)*10))[0]
            multitask_prediction = progressive_learner.predict(
                X=test_x[test_idx], task_id=ii
            )
            base_tasks.append(task+1)
            tasks.append(ii+1)
            accuracies_across_tasks.append(
                np.mean(multitask_prediction == test_y[test_idx])
            )

    df_multitask['task'] = tasks
    df_multitask['base_task'] = base_tasks
    df_multitask['accuracy'] = accuracies_across_tasks

    df_singletask['task'] = list(range(1,num_tasks+1))
    df_singletask['accuracy'] = singletask_accuracy

    summary = (df_multitask, df_singletask)

    with open('results/'+model+'.pickle', 'wb') as f:
        pickle.dump(summary, f)

#%%
if __name__ == "__main__":
    experiment(model='synf')

# %%
