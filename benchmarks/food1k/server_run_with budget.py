import tensorflow as tf
from tensorflow import keras
import numpy as np
from random import sample
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.transformers import (
    NeuralClassificationTransformer,
    TreeClassificationTransformer,
)
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter
from sklearn.model_selection import train_test_split
from tensorflow.keras.backend import clear_session 
import pandas as pd
import pickle

#%%
TRAIN_DATADIR = '/cis/home/jdey4/LargeFineFoodAI/Train'
VAL_DATADIR = '/cis/home/jdey4/LargeFineFoodAI/Val'

CATEGORIES = list(range(20))
SAMPLE_PER_CLASS = 60
NUM_CLASS_PER_TASK = 20
IMG_SIZE = 50

#%%
def get_data(task=0):
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    
    categories_to_consider = range(task*NUM_CLASS_PER_TASK,(task+1)*NUM_CLASS_PER_TASK)
    for category in categories_to_consider:
        path = os.path.join(TRAIN_DATADIR, str(category))

        images = os.listdir(path)
        total_images = len(images)
        train_indx = sample(range(total_images), SAMPLE_PER_CLASS)
        test_indx = np.delete(range(total_images), train_indx)
        for ii in train_indx:
            image_data = cv2.imread(
                    os.path.join(path, images[ii])
                )
            resized_image = cv2.resize(
                image_data, 
                (IMG_SIZE, IMG_SIZE)
            )
            train_X.append(
                resized_image
            )
            train_y.append(
                category
            )
        for ii in test_indx:
            image_data = cv2.imread(
                    os.path.join(path, images[ii])
                )
            resized_image = cv2.resize(
                image_data, 
                (IMG_SIZE, IMG_SIZE)
            )
            test_X.append(
                resized_image
            )
            test_y.append(
                category
            )

    train_X = np.array(train_X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    train_y = np.array(train_y)
    test_X = np.array(test_X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    test_y = np.array(test_y)
    
    return train_X, train_y, test_X, test_y

#%%
def experiment(model='synf', ntrees=10, rep=1, budget=40):
    num_tasks = 50
    tasks = []
    base_tasks = []
    accuracies_across_tasks = []
    singletask_accuracy = []
    df_multitask = pd.DataFrame()
    df_singletask = pd.DataFrame()
    transformers_to_consider = []
    
    if model == "synn":

        clear_session()  # clear GPU memory before each run, to avoid OOM error

        default_transformer_class = NeuralClassificationTransformer

        network = keras.Sequential()
        network.add(
            layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(IMG_SIZE,IMG_SIZE,3),
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
        network.add(layers.Dense(units=NUM_CLASS_PER_TASK, activation="softmax"))  # units=10

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
        default_voter_kwargs = {"k": int(np.log2(500))}
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
    
    test_x_task = []
    test_y_task = []
    for task in range(num_tasks):
        print("doing task ", task)
        
        if task > budget-1:
            transformers_to_consider.pop(0)
            
        transformers_to_consider.append(task)
        train_x, train_y, test_x, test_y = get_data(task)
        #train_x = train_x.reshape(-1, 3*IMG_SIZE*IMG_SIZE)
        #test_x = test_x.reshape(-1, 3*IMG_SIZE*IMG_SIZE)
        
        test_x_task.append(
            test_x
        )
        test_y_task.append(
            test_y
        )
        progressive_learner.add_task(
            X=train_x,
            y=train_y,
            task_id=task,
            num_transformers=1 if model == "synn" else ntrees,
            transformer_voter_decider_split=[0.67, 0.33, 0],
            decider_kwargs={"classes": np.unique(train_y)},
        )

        singletask_prediction = progressive_learner.predict(
            X=test_x, transformer_ids=[task], task_id=task
        )
        singletask_accuracy.append(
            np.mean(singletask_prediction==test_y)
        )
        print('accuracy ',np.mean(singletask_prediction==test_y))
        for ii in range(task+1):
            multitask_prediction = progressive_learner.predict(
                X=test_x_task[ii], transformer_ids=transformers_to_consider, task_id=ii
            )
            acc = np.mean(multitask_prediction==test_y_task[ii])
            print('task ',ii,' accuracy ', acc)
            base_tasks.append(task+1)
            tasks.append(ii+1)
            accuracies_across_tasks.append(
                np.mean(multitask_prediction == test_y_task[ii])
            )

    df_multitask['task'] = tasks
    df_multitask['base_task'] = base_tasks
    df_multitask['accuracy'] = accuracies_across_tasks

    df_singletask['task'] = list(range(1,num_tasks+1))
    df_singletask['accuracy'] = singletask_accuracy

    summary = (df_multitask, df_singletask)

    with open('results/'+model+'_'+str(rep)+'_' + str(budget)+'.pickle', 'wb') as f:
        pickle.dump(summary, f)


#%%
reps = 1
budgets = [10,20,30,40]

for budget in budgets:
    for ii in range(reps):
        experiment(model='synn',rep=ii, budget=budget)