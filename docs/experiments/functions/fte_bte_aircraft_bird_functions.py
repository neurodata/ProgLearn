import random

import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras import layers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.transformers import NeuralClassificationTransformer, TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter
from sklearn.model_selection import train_test_split
from tensorflow.keras.backend import clear_session  # To avoid OOM error when using odin


def load_tasks(path_aircraft_x_all, path_aircraft_y_all, path_birdsnap_x_all, path_birdsnap_y_all):
    n_task_aircraft = 5  # Set up 5 tasks from aircraft
    n_task_birdsnap = 5  # Set up 5 tasks from birdsnap
    n_label_per_task = 20
    n_sample_label = 90  # Because a large proportion of the classes in Birdsnap contains less than 90 samples
    test_size = 0.4  # 54 train, 36 test
    # Load Data
    aircraft_x_all = np.load(path_aircraft_x_all)
    aircraft_y_all = np.load(path_aircraft_y_all)
    birdsnap_x_all = np.load(path_birdsnap_x_all)
    birdsnap_y_all = np.load(path_birdsnap_y_all)
    # Split into tasks
    train_x_task = [[], [], [], [], [], [], [], [], [], []]
    test_x_task = [[], [], [], [], [], [], [], [], [], []]
    train_y_task = [[], [], [], [], [], [], [], [], [], []]
    test_y_task = [[], [], [], [], [], [], [], [], [], []]
    # load aircraft data to tasks
    for i in range(n_task_aircraft):
        X = np.empty([0, 32, 32, 3])
        Y = np.empty([0, ])
        for j in range(n_label_per_task):
            new_x = aircraft_x_all[aircraft_y_all == n_label_per_task * i + j + 1][:n_sample_label]
            new_y = aircraft_y_all[aircraft_y_all == n_label_per_task * i + j + 1][:n_sample_label]
            X = np.concatenate((X, new_x), axis=0)
            Y = np.concatenate((Y, new_y), axis=0)
        train_x_task[i], test_x_task[i], train_y_task[i], test_y_task[i] = train_test_split(X, Y, test_size=test_size)
    # load birdsnap data to tasks
    for i in range(n_task_birdsnap):
        X = np.empty([0, 32, 32, 3])
        Y = np.empty([0, ])
        for j in range(n_label_per_task):
            new_x = birdsnap_x_all[birdsnap_y_all == n_label_per_task * i + j + 1][:n_sample_label]
            new_y = birdsnap_y_all[birdsnap_y_all == n_label_per_task * i + j + 1][:n_sample_label] + 100
            X = np.concatenate((X, new_x), axis=0)
            Y = np.concatenate((Y, new_y), axis=0)
        train_x_task[n_task_aircraft + i], test_x_task[n_task_aircraft + i], train_y_task[n_task_aircraft + i], \
        test_y_task[n_task_aircraft + i] = train_test_split(X, Y, test_size=test_size)
    return train_x_task, test_x_task, train_y_task, test_y_task


def show_image(train_x_task):
    # task 1-5 is aircraft, 6-10 is birdsnap
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    img = train_x_task[random.randint(0, 4)][random.randint(0, 89)]
    plt.imshow(img.astype(np.uint8))
    plt.xticks([])
    plt.yticks([])
    plt.title("Sample in Aircraft")

    plt.subplot(1, 2, 2)
    img = train_x_task[random.randint(5, 9)][random.randint(0, 89)]
    plt.imshow(img.astype(np.uint8))
    plt.xticks([])
    plt.yticks([])
    plt.title("Sample in Birdsnap")


def single_experiment(train_x_task, test_x_task, train_y_task, test_y_task, ntrees=10, model='odif'):
    num_tasks = 10
    num_points_per_task = 1800
    accuracies = np.zeros(65, dtype=float)

    if model == 'odin':

        clear_session()  # clear GPU memory before each run, to avoid OOM error

        default_transformer_class = NeuralClassificationTransformer

        network = keras.Sequential()
        network.add(
            layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=np.shape(train_x_task[0])[1:]))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding="same", activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding="same", activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding="same", activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=254, kernel_size=(3, 3), strides=2, padding="same", activation='relu'))

        network.add(layers.Flatten())
        network.add(layers.BatchNormalization())
        network.add(layers.Dense(2000, activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Dense(2000, activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Dense(units=20, activation='softmax'))  # units=10

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
        default_voter_kwargs = {"k": int(np.log2(num_points_per_task))}
        default_decider_class = SimpleArgmaxAverage

    elif model == 'odif':
        for i in range(num_tasks):
            train_x_task[i] = train_x_task[i].reshape(1080, -1)
            test_x_task[i] = test_x_task[i].reshape(720, -1)

        default_transformer_class = TreeClassificationTransformer
        default_transformer_kwargs = {"kwargs": {"max_depth": 30}}
        default_voter_class = TreeClassificationVoter
        default_voter_kwargs = {}
        default_decider_class = SimpleArgmaxAverage

    progressive_learner = ProgressiveLearner(default_transformer_class=default_transformer_class,
                                             default_transformer_kwargs=default_transformer_kwargs,
                                             default_voter_class=default_voter_class,
                                             default_voter_kwargs=default_voter_kwargs,
                                             default_decider_class=default_decider_class)

    for i in range(num_tasks):
        progressive_learner.add_task(
            X=train_x_task[i],
            y=train_y_task[i],
            task_id=i,
            num_transformers=1 if model == "odin" else ntrees,
            transformer_voter_decider_split=[0.67, 0.33, 0],
            decider_kwargs={"classes": np.unique(train_y_task[i])},
        )
        prediction = progressive_learner.predict(
            X=test_x_task[i], transformer_ids=[i], task_id=i
        )
        accuracies[i] = np.mean(prediction == test_y_task[i])

        for j in range(num_tasks):
            if j > i:
                pass  # this is not wrong but misleading, should be continue
            else:
                odif_predictions = progressive_learner.predict(test_x_task[j], task_id=j)

            accuracies[10 + j + (i * (i + 1)) // 2] = np.mean(odif_predictions == test_y_task[j])
    # print('single experiment done!')

    return accuracies


def calculate_results(accuracy_all):
    num_tasks = 10
    err = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            err[i].append(1 - accuracy_all[10 + ((j + i) * (j + i + 1)) // 2 + i])

    bte = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            bte[i].append(err[i][0] / err[i][j])

    fte = np.zeros(10, dtype=float)
    for i in range(num_tasks):
        fte[i] = (1 - accuracy_all[i]) / err[i][0]

    te = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            te[i].append((1 - accuracy_all[i]) / err[i][j])

    return err, bte, fte, te


def plot_all(err, bte, fte, te):
    num_tasks = 10
    sns.set(style='ticks')

    fontsize = 30
    ticksize = 24
    markersize = 16
    linewidth = 3

    fig, ax = plt.subplots(2, 2, figsize=(20, 14.5))

    # FTE
    ax[0][0].axvline(x=5, linestyle="dashed", color='black', alpha=0.5)
    ax[0][0].text(5, 1.001, 'Dataset Switch', fontsize=25, color='black')
    ax[0][0].plot(np.arange(1, num_tasks + 1), fte, c='red', marker='.', markersize=markersize, linewidth=linewidth,
                  label='Odif')

    ax[0][0].hlines(1, 1, num_tasks, colors='grey', linestyles='dashed', linewidth=0.5 * linewidth)
    ax[0][0].tick_params(labelsize=ticksize)
    ax[0][0].set_xlabel('Number of tasks seen', fontsize=fontsize)
    ax[0][0].set_ylabel('log FTE', fontsize=fontsize)
    ax[0][0].set_yticks([1, 1.0408, 1.0833])
    ax[0][0].set_ylim(0.99, 1.09)

    log_lbl = np.round(
        np.log([1, 1.0408, 1.0833]),
        2
    )
    labels = [item.get_text() for item in ax[0][0].get_yticklabels()]

    for ii, _ in enumerate(labels):
        labels[ii] = str(log_lbl[ii])

    ax[0][0].set_yticklabels(labels)

    # BTE
    ax[0][1].axvline(x=5, linestyle="dashed", color='black', alpha=0.5)
    for i in range(num_tasks):
        et = np.asarray(bte[i])
        ns = np.arange(i + 1, num_tasks + 1)
        ax[0][1].plot(ns, et, c='red', linewidth=linewidth)

    ax[0][1].set_xlabel('Number of tasks seen', fontsize=fontsize)
    ax[0][1].set_ylabel('log BTE', fontsize=fontsize)
    ax[0][1].tick_params(labelsize=ticksize)
    ax[0][1].hlines(1, 1, num_tasks, colors='grey', linestyles='dashed', linewidth=0.5 * linewidth)

    ax[0][1].set_yticks([1, 1.0202, 1.0408])
    ax[0][1].set_ylim(0.995, 1.048)

    log_lbl = np.round(
        np.log([1, 1.0202, 1.0408]),
        2
    )
    labels = [item.get_text() for item in ax[0][1].get_yticklabels()]

    for ii, _ in enumerate(labels):
        labels[ii] = str(log_lbl[ii])

    ax[0][1].set_yticklabels(labels)

    # TE
    ax[1][0].axvline(x=5, linestyle="dashed", color='black', alpha=0.5)
    for i in range(num_tasks):
        et = np.asarray(te[i])
        ns = np.arange(i + 1, num_tasks + 1)
        ax[1][0].plot(ns, et, c='red', linewidth=linewidth)

    ax[1][0].set_xlabel('Number of tasks seen', fontsize=fontsize)
    ax[1][0].set_ylabel('log Transfer Efficiency', fontsize=fontsize)
    ax[1][0].tick_params(labelsize=ticksize)
    ax[1][0].hlines(1, 1, num_tasks, colors='grey', linestyles='dashed', linewidth=0.5 * linewidth)

    ax[1][0].set_yticks([1, 1.0513, 1.1052])
    ax[1][0].set_ylim(0.99, 1.12)

    log_lbl = np.round(
        np.log([1, 1.0513, 1.1052]),
        2
    )
    labels = [item.get_text() for item in ax[1][0].get_yticklabels()]

    for ii, _ in enumerate(labels):
        labels[ii] = str(log_lbl[ii])

    ax[1][0].set_yticklabels(labels)

    # Accuracy
    ax[1][1].axvline(x=5, linestyle="dashed", color='black', alpha=0.5)
    for i in range(num_tasks):
        et = 1 - np.asarray(err[i])
        ns = np.arange(i + 1, num_tasks + 1)
        if i == 0:
            ax[1][1].plot(ns, et, c='red', linewidth=linewidth, label='Odif')
        else:
            ax[1][1].plot(ns, et, c='red', linewidth=linewidth)

    ax[1][1].set_xlabel('Number of tasks seen', fontsize=fontsize)
    ax[1][1].set_ylabel('Accuracy', fontsize=fontsize)
    ax[1][1].tick_params(labelsize=ticksize)
