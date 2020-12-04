import os
import cv2
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import time

from sklearn.model_selection import train_test_split

import keras
from keras import layers

from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import (
    TreeClassificationTransformer,
    NeuralClassificationTransformer,
)
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter


def load_spoken_digit(path_recordings):
    file = os.listdir(path_recordings)

    AudioData=[]       # audio data
    X_spec = []        # STFT spectrogram
    X_spec_mini = []   # resized image, 28*28
    Y_number = []      # label of number
    Y_speaker = []     # label of speaker

    for i in file:
        x , sr = librosa.load('D:/Python Exploration/free-spoken-digit-dataset/recordings/'+i, sr = 8000) # path of the audio files
        X = librosa.stft(x,n_fft = 128) # STFT
        Xdb = librosa.amplitude_to_db(abs(X)) # Convert an amplitude spectrogram to dB-scaled spectrogram
        Xdb_28 = cv2.resize(Xdb, (28, 28)) # resize into 28 by 28
        y_n = i[0] # number
        y_s = i[2] # first letter of speaker's name

        AudioData.append(x)
        X_spec.append(Xdb)
        X_spec_mini.append(Xdb_28)
        Y_number.append(y_n)
        Y_speaker.append(y_s)

    X_spec_mini = np.array(X_spec_mini)
    Y_number    = np.array(Y_number).astype(int)
    Y_speaker   = np.array(Y_speaker)

    return AudioData , X_spec_mini, Y_number, Y_speaker


def play_audio(num, AudioData, Y_number, Y_speaker):
    print('This is number',Y_number[num],'spoken by speaker',Y_speaker[num].upper(),'\nDuration:',AudioData[num].shape[0],'samples in',AudioData[num].shape[0]/8000,'seconds')
    ipd.Audio(AudioData[num], rate=8000)


def display_spectrogram(X_spec_mini, Y_number, Y_speaker, num):

    Label_speaker = ['g', 'j', 'l', 'n', 't', 'y'] # first letter of speaker's name
    fig, axes = plt.subplots(nrows=6, ncols=8, sharex=True, sharey=True, figsize=(12, 9))
    for j, speaker in enumerate(Label_speaker):
        opt = np.where((Y_speaker == speaker) & (Y_number == num))[0]
        ch = np.random.choice(opt,8,replace=False) #randomly choose 8 different samples to display

        for i in range(8):
            axes[j,i].imshow(X_spec_mini[ch[i]], cmap='hot', interpolation='nearest')
            axes[j,i].axis('off')
            if i == 0: axes[j,i].text(-9, 12, 'speaker '+speaker.upper(), size=14, va='center', rotation='vertical')
            else: continue

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.05,
                        wspace=0.05)
    plt.suptitle("Short-Time Fourier Transform Spectrograms of Number "+str(num), fontsize=18)


def run_experiment(X_all, Y_all, Y_all_speaker, ntrees=19, model = 'uf', num_repetition = 1):
    num_tasks = 6
    speakers = ['g', 'j', 'l', 'n', 't', 'y']
    accuracies_across_tasks = []

    if model == 'uf':
        default_transformer_class = TreeClassificationTransformer
        default_transformer_kwargs = {"kwargs": {"max_depth": 30}}
        default_voter_class = TreeClassificationVoter
        default_voter_kwargs = {}
        default_decider_class = SimpleArgmaxAverage
    elif model == 'dnn':
        default_transformer_class = NeuralClassificationTransformer

        network = keras.Sequential()
        network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=np.shape(X_all)[1:])) ##############################################
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

        default_transformer_kwargs = {"network" : network,
                                      "euclidean_layer_idx" : -2,
                                      "num_classes" : 10,
                                      "optimizer" : keras.optimizers.Adam(3e-4)
                                     }

        default_voter_class = KNNClassificationVoter
        #default_voter_kwargs = {"k" : int(np.log2(num_points_per_task))}####################################################################################
        default_voter_kwargs = {"k" : int(np.log2(500))}

        default_decider_class = SimpleArgmaxAverage

    for i in range(num_repetition):
        np.random.shuffle(speakers)
        print(i+1,'time repetition')##########
        for j , speaker in enumerate(speakers):
            # initialization
            progressive_learner = ProgressiveLearner(
                default_transformer_class=default_transformer_class,
                default_transformer_kwargs=default_transformer_kwargs,
                default_voter_class=default_voter_class,
                default_voter_kwargs=default_voter_kwargs,
                default_decider_class=default_decider_class,
            )
            #print('task 0 speaker is ',speaker)############
            index = np.where(Y_all_speaker==speaker)
            X = X_all[index]
            Y = Y_all[index]
            train_x_task0, test_x_task0, train_y_task0, test_y_task0 = train_test_split(X, Y, test_size=0.25)
            progressive_learner.add_task(
                X=train_x_task0,
                y=train_y_task0,
                task_id = 0,
                num_transformers= 1 if model == "dnn" else ntrees,
                transformer_voter_decider_split=[0.67, 0.33, 0],
                decider_kwargs={"classes": np.unique(train_y_task0)},
            )
            task_0_predictions = progressive_learner.predict(test_x_task0, task_id=0)
            accuracies_across_tasks.append(np.mean(task_0_predictions == test_y_task0))

            for k , speaker in enumerate(speakers):
                if k == j:
                    pass
                else:
                    index = np.where(Y_all_speaker==speaker)
                    X_train = X_all[index]
                    Y_train = Y_all[index]
                    progressive_learner.add_transformer(
                        X=X_train,
                        y=Y_train,
                        transformer_data_proportion=1,
                        num_transformers= ntrees,
                        backward_task_ids=[0],
                    )
                task_0_predictions = progressive_learner.predict(test_x_task0, task_id=0)
                accuracies_across_tasks.append(np.mean(task_0_predictions == test_y_task0))

    # Average the results
    accuracy_all_task = np.array(accuracies_across_tasks).reshape((num_repetition,num_tasks*(num_tasks+1)))
    accuracy_all_task = np.average(accuracy_all_task, axis = 0)

    return accuracy_all_task

def calculate_results(accuracy_all_task):
    num_tasks = 6
    acc = [[] for i in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            acc[i].append(accuracy_all_task[7*i+i+1+j])

    bte = [[] for i in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            err_taskt_only  = 1-accuracy_all_task[7*i]
            err_up_to_taskt = 1-accuracy_all_task[7*i+i+1]
            err_all_seen    = 1-accuracy_all_task[7*i+i+1+j]
            bte[i].append(err_up_to_taskt/err_all_seen)

    fte = [[] for i in range(num_tasks)]
    for i in range(num_tasks):
        err_taskt_only  = 1-accuracy_all_task[7*i]
        err_up_to_taskt = 1-accuracy_all_task[7*i+i+1]
        fte[i].append(err_taskt_only/err_up_to_taskt)

    te = [[]for i in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks - i):
            err_taskt_only  = 1-accuracy_all_task[7*i]
            err_all_seen    = 1-accuracy_all_task[7*i+i+1+j]
            te[i].append(err_taskt_only / err_all_seen)

    return acc, bte, fte, te

def plot_results(acc, bte, fte, te):
    num_tasks = 6
    sns.set()
    clr = ["#e41a1c", "#a65628", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#CCCC00"]
    fontsize=22
    ticksize=20

    fig, ax = plt.subplots(2,2, figsize=(16,11.5))
    ax[0][0].plot(np.arange(1,num_tasks+1), fte, c='red', marker='.', markersize=14, linewidth=3)
    ax[0][0].hlines(1, 1,num_tasks, colors='grey', linestyles='dashed',linewidth=1.5)
    ax[0][0].tick_params(labelsize=ticksize)
    ax[0][0].set_xlabel('Number of tasks seen', fontsize=fontsize)
    ax[0][0].set_ylabel('FTE', fontsize=fontsize)

    for i in range(num_tasks):
        et = np.asarray(bte[i])
        ns = np.arange(i + 1, num_tasks + 1)
        ax[0][1].plot(ns, et, c='red', linewidth = 2.6)

    ax[0][1].set_xlabel('Number of tasks seen', fontsize=fontsize)
    ax[0][1].set_ylabel('BTE', fontsize=fontsize)
    ax[0][1].tick_params(labelsize=ticksize)
    ax[0][1].hlines(1, 1,num_tasks, colors='grey', linestyles='dashed',linewidth=1.5)

    for i in range(num_tasks):
        et = np.asarray(te[i])
        ns = np.arange(i + 1, num_tasks + 1)
        ax[1][0].plot(ns, et, c='red', linewidth = 2.6)

    ax[1][0].set_xlabel('Number of tasks seen', fontsize=fontsize)
    ax[1][0].set_ylabel('Transfer Efficiency', fontsize=fontsize)
    #ax[1][0].set_xticks(np.arange(1,10))
    ax[1][0].tick_params(labelsize=ticksize)
    ax[1][0].hlines(1, 1,num_tasks, colors='grey', linestyles='dashed',linewidth=1.5)

    for i in range(num_tasks):
        et = np.asarray(acc[i])
        ns = np.arange(i + 1, num_tasks + 1)
        ax[1][1].plot(ns, et , c='red', linewidth = 2.6)

    ax[1][1].set_xlabel('Number of tasks seen', fontsize=fontsize)
    ax[1][1].set_ylabel('Accuracy', fontsize=fontsize)
    ax[1][1].tick_params(labelsize=ticksize)
