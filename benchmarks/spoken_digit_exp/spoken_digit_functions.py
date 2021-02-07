import os
import time

import IPython.display as ipd
import cv2
import keras
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import save, load
from keras import layers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.transformers import NeuralClassificationTransformer, TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter
from sklearn.model_selection import train_test_split

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


def single_experiment(x, y, y_speaker, ntrees=20, model='uf', shuffle=False):
    num_tasks = 6
    num_points_per_task = 3000 / num_tasks
    speakers = ['g', 'j', 'l', 'n', 't', 'y']
    accuracies_across_tasks = []

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

    for j, task0_speaker in enumerate(speakers):
        progressive_learner = ProgressiveLearner(default_transformer_class=default_transformer_class,
                                                 default_transformer_kwargs=default_transformer_kwargs,
                                                 default_voter_class=default_voter_class,
                                                 default_voter_kwargs=default_voter_kwargs,
                                                 default_decider_class=default_decider_class)

        index = np.where(y_speaker == task0_speaker)
        x_task0 = x_all[index]
        y_task0 = y_all[index]
        train_x_task0, test_x_task0, train_y_task0, test_y_task0 = train_test_split(x_task0, y_task0,
                                                                                    test_size=0.25)
        progressive_learner.add_task(
            X=train_x_task0,
            y=train_y_task0,
            task_id=0,
            num_transformers=1 if model == "dnn" else ntrees,
            transformer_voter_decider_split=[0.67, 0.33, 0],
            decider_kwargs={"classes": np.unique(train_y_task0)},
        )
        task_0_predictions = progressive_learner.predict(test_x_task0, task_id=0)
        accuracies_across_tasks.append(np.mean(task_0_predictions == test_y_task0))

        for k, contribute_speaker in enumerate(speakers):
            if k == j:
                pass
            else:
                index = np.where(y_speaker == contribute_speaker)
                x_train = x_all[index]
                y_train = y_all[index]
                progressive_learner.add_transformer(
                    X=x_train,
                    y=y_train,
                    transformer_data_proportion=1,
                    num_transformers=1 if model == "dnn" else ntrees,
                    backward_task_ids=[0],
                )
            task_0_predictions = progressive_learner.predict(test_x_task0, task_id=0)
            accuracies_across_tasks.append(np.mean(task_0_predictions == test_y_task0))
    
    current_time = time.strftime("%H%M%S",time.localtime())
    filename = 'result/cache/accuracies_ntree'+str(ntrees)+'_'+model+current_time+'.npy'
    save(filename, accuracies_across_tasks)

    return accuracies_across_tasks

    
def calculate_results(accuracy_list):
    """
    We have 6 tasks, each task generates 7 accuracies (among which 1 is redundant) in order to compute bte,
    fte at each points, so there are 42 points in total.

    This may help:

    Position of Task0-only accuracy
    '
    = = - - - - -
    - = = - - - -
    - - = = - - -
    - - - = = - -
    - - - - = = -
    - - - - - = =
    """
    accuracy_all = np.array([i for i in accuracy_list])
    accuracy_all = np.average(accuracy_all, axis = 0)
    
    num_tasks = 6
    acc = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            acc[i].append(accuracy_all[7 * i + i + 1 + j])  # "7*i" first of each row, "+(i+1)" skip to the second "="

    bte = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            err_up_to_taskt = 1 - accuracy_all[7 * i + i + 1]
            err_all_seen = 1 - accuracy_all[7 * i + i + 1 + j]  # "7*i + i+1" is at the second "="
            bte[i].append(err_up_to_taskt / err_all_seen)

    fte = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        err_taskt_only = 1 - accuracy_all[7 * i]
        err_up_to_taskt = 1 - accuracy_all[7 * i + i + 1]
        fte[i].append(err_taskt_only / err_up_to_taskt)

    te = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            err_taskt_only = 1 - accuracy_all[7 * i]
            err_all_seen = 1 - accuracy_all[7 * i + i + 1 + j]
            te[i].append(err_taskt_only / err_all_seen)

    return acc, bte, fte, te


def plot_results(acc, bte, fte, te, filename):
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
        et = np.asarray(acc[i])
        ns = np.arange(i + 1, num_tasks + 1)
        ax[1][1].plot(ns, et, c='red', linewidth=2.6)

    ax[1][1].set_xlabel('Number of tasks seen', fontsize=fontsize)
    ax[1][1].set_ylabel('Accuracy', fontsize=fontsize)
    ax[1][1].tick_params(labelsize=ticksize)
    
    plt.savefig(filename, dpi=300)