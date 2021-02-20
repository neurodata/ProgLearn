import os

import IPython.display as ipd
import cv2
import keras
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras import layers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from proglearn.deciders import SimpleArgmaxAverage
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.transformers import NeuralClassificationTransformer, TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter

from tensorflow.keras.backend import clear_session # To avoid OOM error when using dnn


def load_spoken_digit(path_recordings):
    file = os.listdir(path_recordings)

    audio_data = []  # audio data
    x_spec = []  # STFT spectrogram
    x_spec_mini = []  # resized image, 28*28
    y_number = []  # label of number
    y_speaker = []  # label of speaker

    for i in file:
        x, sr = librosa.load(path_recordings + i, sr=8000)
        x_stft = librosa.stft(x, n_fft=128)  # Extract STFT
        x_stft_db = librosa.amplitude_to_db(abs(x_stft))  # Convert an amplitude spectrogram to dB-scaled spectrogram
        x_stft_db_mini = cv2.resize(x_stft_db, (28, 28))  # Resize into 28 by 28
        y_n = i[0]  # number
        y_s = i[2]  # first letter of speaker's name

        audio_data.append(x)
        x_spec.append(x_stft_db)
        x_spec_mini.append(x_stft_db_mini)
        y_number.append(y_n)
        y_speaker.append(y_s)

    x_spec_mini = np.array(x_spec_mini)
    y_number = np.array(y_number).astype(int)
    y_speaker = np.array(y_speaker)

    return audio_data, x_spec_mini, y_number, y_speaker


def play_audio(num, audio_data, y_number, y_speaker):
    print('This is number', y_number[num], 'spoken by speaker', y_speaker[num].upper(), '\nDuration:',
          audio_data[num].shape[0], 'samples in', audio_data[num].shape[0] / 8000, 'seconds')
    ipd.Audio(audio_data[num], rate=8000)


def display_spectrogram(x_spec_mini, y_number, y_speaker, num):
    label_speaker = ['g', 'j', 'l', 'n', 't', 'y']  # first letter of speaker's name
    fig, axes = plt.subplots(nrows=6, ncols=8, sharex=True, sharey=True, figsize=(12, 9))
    for j, speaker in enumerate(label_speaker):
        opt = np.where((y_speaker == speaker) & (y_number == num))[0]
        ch = np.random.choice(opt, 8, replace=False)  # randomly choose 8 different samples from the speaker to display

        for i in range(8):
            axes[j, i].imshow(x_spec_mini[ch[i]], cmap='hot', interpolation='nearest')
            axes[j, i].axis('off')
            if i == 0:
                axes[j, i].text(-9, 12, 'speaker ' + speaker.upper(), size=14, va='center', rotation='vertical')
            else:
                continue

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.05,
                        wspace=0.05)
    plt.suptitle("Short-Time Fourier Transform Spectrogram of Number " + str(num), fontsize=18)


def single_experiment(x, y, y_speaker, ntrees=10, model='uf', shuffle=False):
    num_tasks = 6
    num_points_per_task = 3000 / num_tasks
    speakers = ['g', 'j', 'l', 'n', 't', 'y']
    single_task_accuracies = np.zeros(num_tasks, dtype=float)
    accuracies = np.zeros(27, dtype=float)

    if model == 'dnn':
        x_all = x
        y_all = y

        clear_session() # clear GPU memory before each run, to avoid OOM error

        default_transformer_class = NeuralClassificationTransformer

        network = keras.Sequential()
        network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=np.shape(x_all)[1:]))
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
        network.add(layers.Dense(units=10, activation='softmax'))

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

    elif model == 'uf':
        x_all = x.reshape(3000, -1)
        y_all = y

        default_transformer_class = TreeClassificationTransformer
        default_transformer_kwargs = {"kwargs": {"max_depth": 30}}
        default_voter_class = TreeClassificationVoter
        default_voter_kwargs = {}
        default_decider_class = SimpleArgmaxAverage

    if shuffle:
        np.random.shuffle(speakers)
    else:
        pass

    progressive_learner = ProgressiveLearner(default_transformer_class=default_transformer_class,
                                             default_transformer_kwargs=default_transformer_kwargs,
                                             default_voter_class=default_voter_class,
                                             default_voter_kwargs=default_voter_kwargs,
                                             default_decider_class=default_decider_class)

    train_x_task, test_x_task, train_y_task, test_y_task = [[], [], [], [], [], []], [[], [], [], [], [], []], [[], [],
                                                                                                                [], [],
                                                                                                                [],
                                                                                                                []], [
                                                               [], [], [], [], [], []]

    for j, task0_speaker in enumerate(speakers):
        index = np.where(y_speaker == task0_speaker)
        x_task0 = x_all[index]
        y_task0 = y_all[index]
        train_x_task[j], test_x_task[j], train_y_task[j], test_y_task[j] = train_test_split(x_task0, y_task0,
                                                                                            test_size=0.45)

        # print(train_y_task[j], train_x_task[j].shape)
        progressive_learner.add_task(
            X=train_x_task[j],
            y=train_y_task[j],
            task_id=j,
            num_transformers=1 if model == "dnn" else ntrees,
            transformer_voter_decider_split=[0.67, 0.33, 0],
            decider_kwargs={"classes": np.unique(train_y_task[j])},
        )
        uf_predictions = progressive_learner.predict(
            X=test_x_task[j], transformer_ids=[j], task_id=j
        )
        accuracies[j] = np.mean(uf_predictions == test_y_task[j])

        for k, contribute_speaker in enumerate(speakers):
            if k > j:
                pass
            else:
                odif_predictions = progressive_learner.predict(test_x_task[k], task_id=k)

            accuracies[6 + k + (j * (j + 1)) // 2] = np.mean(odif_predictions == test_y_task[k])

    return accuracies


def calculate_results(accuracy_all):
    num_tasks = 6
    err = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            err[i].append(1 - accuracy_all[6 + ((j + i) * (j + i + 1)) // 2 + i])

    bte = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            bte[i].append(err[i][0] / err[i][j])

    fte = np.zeros(6, dtype=float)
    for i in range(num_tasks):
        fte[i] = (1 - accuracy_all[i]) / err[i][0]

    te = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            te[i].append((1 - accuracy_all[i]) / err[i][j])

    return err, bte, fte, te


def plot_results(acc, bte, fte, te):
    num_tasks = 6
    sns.set()
    clr = ["#e41a1c", "#a65628", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#CCCC00"]
    fontsize = 22
    ticksize = 20

    fig, ax = plt.subplots(2, 2, figsize=(16, 11.5))
    ax[0][0].plot(np.arange(1, num_tasks + 1), fte, c='red', marker='.', markersize=14, linewidth=3)
    ax[0][0].hlines(1, 1, num_tasks, colors='grey', linestyles='dashed', linewidth=1.5)
    ax[0][0].tick_params(labelsize=ticksize)
    ax[0][0].set_xlabel('Number of tasks seen', fontsize=fontsize)
    ax[0][0].set_ylabel('FTE', fontsize=fontsize)

    for i in range(num_tasks):
        et = np.asarray(bte[i])
        ns = np.arange(i + 1, num_tasks + 1)
        ax[0][1].plot(ns, et, c='red', linewidth=2.6)

    ax[0][1].set_xlabel('Number of tasks seen', fontsize=fontsize)
    ax[0][1].set_ylabel('BTE', fontsize=fontsize)
    ax[0][1].tick_params(labelsize=ticksize)
    ax[0][1].hlines(1, 1, num_tasks, colors='grey', linestyles='dashed', linewidth=1.5)

    for i in range(num_tasks):
        et = np.asarray(te[i])
        ns = np.arange(i + 1, num_tasks + 1)
        ax[1][0].plot(ns, et, c='red', linewidth=2.6)

    ax[1][0].set_xlabel('Number of tasks seen', fontsize=fontsize)
    ax[1][0].set_ylabel('Transfer Efficiency', fontsize=fontsize)
    # ax[1][0].set_xticks(np.arange(1,10))
    ax[1][0].tick_params(labelsize=ticksize)
    ax[1][0].hlines(1, 1, num_tasks, colors='grey', linestyles='dashed', linewidth=1.5)

    for i in range(num_tasks):
        et = 1 - np.asarray(acc[i])
        ns = np.arange(i + 1, num_tasks + 1)
        ax[1][1].plot(ns, et, c='red', linewidth=2.6)

    ax[1][1].set_xlabel('Number of tasks seen', fontsize=fontsize)
    ax[1][1].set_ylabel('Accuracy', fontsize=fontsize)
    ax[1][1].tick_params(labelsize=ticksize)


def plot_paper_figure(bte, fte):
    sns.set_context("talk")
    num_tasks = 6

    clr = ["#e41a1c", "#a65628", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#CCCC00"]
    fontsize = 22
    ticksize = 20

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].plot(np.arange(1, num_tasks + 1), fte, c='red', marker='.', markersize=14, linewidth=3)
    ax[0].hlines(1, 1, num_tasks, colors='grey', linestyles='dashed', linewidth=1.5)
    ax[0].tick_params(labelsize=ticksize)
    ax[0].set_xlabel('Number of tasks seen', fontsize=fontsize)
    ax[0].set_ylabel('log Forward TE', fontsize=fontsize)
    ax[0].set_yticks([1, 1.25, 1.5])

    log_lbl = np.round(
        np.log([1, 1.25, 1.5]),
        2
    )
    labels = [item.get_text() for item in ax[0].get_yticklabels()]

    for ii, _ in enumerate(labels):
        labels[ii] = str(log_lbl[ii])

    ax[0].set_yticklabels(labels)

    right_side = ax[0].spines["right"]
    right_side.set_visible(False)
    top_side = ax[0].spines["top"]
    top_side.set_visible(False)

    for i in range(num_tasks):
        et = np.asarray(bte[i])
        ns = np.arange(i + 1, num_tasks + 1)
        ax[1].plot(ns, et, c='red', linewidth=2.6)

    ax[1].set_xlabel('Number of tasks seen', fontsize=fontsize)
    ax[1].set_ylabel('log Backward TE', fontsize=fontsize)
    ax[1].tick_params(labelsize=ticksize)
    ax[1].hlines(1, 1, num_tasks, colors='grey', linestyles='dashed', linewidth=1.5)
    ax[1].set_yticks([1, 1.35, 1.7])

    log_lbl = np.round(
        np.log([1, 1.35, 1.7]),
        2
    )
    labels = [item.get_text() for item in ax[1].get_yticklabels()]

    for ii, _ in enumerate(labels):
        labels[ii] = str(log_lbl[ii])

    ax[1].set_yticklabels(labels)

    right_side = ax[1].spines["right"]
    right_side.set_visible(False)
    top_side = ax[1].spines["top"]
    top_side.set_visible(False)

    plt.savefig('language.pdf')