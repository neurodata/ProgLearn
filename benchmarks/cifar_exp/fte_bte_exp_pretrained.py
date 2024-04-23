#%%
import random
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

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import NeuralClassificationTransformer, TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter
from keras.applications import VGG19,ResNet50
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

import tensorflow as tf

import time
import sys
#%%
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-4
    if epoch > 18:
        lr *= 0.5e-3
    elif epoch > 16:
        lr *= 1e-3
    elif epoch > 8:
        lr *= 1e-2
    elif epoch > 4:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

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

#%%
input_shape = (32, 32, 3)

def LF_experiment(train_x, train_y, test_x, test_y, ntrees, shift, slot, model, num_points_per_task, acorn=None):

    df = pd.DataFrame()
    single_task_accuracies = np.zeros(10,dtype=float)
    shifts = []
    tasks = []
    base_tasks = []
    accuracies_across_tasks = []
    train_times_across_tasks = []
    single_task_inference_times_across_tasks = []
    multitask_inference_times_across_tasks = []

    data_augmentation = keras.Sequential(
        [
            layers.Resizing(224,224),
            layers.Normalization()
        ]
    )
    if model == "dnn":
        default_transformer_class = NeuralClassificationTransformer

        network = keras.Sequential()
        base_model_1 = ResNet50(include_top=False,weights=None,input_shape=(32,32,3),classes=y_train.shape[1])
        network.add(base_model_1) #Adds the base model (in this case vgg19 to model_1)
        network.add(layers.Flatten())
        #network.add(layers.Dense(1024,activation=('relu'),input_dim=512))
        network.add(layers.Dense(128,activation=('relu'),input_dim=512)) 
        #network.add(layers.Dense(256,activation=('relu'))) 
        #model_1.add(Dropout(.3))#Adding a dropout layer that will randomly drop 30% of the weights
        #network.add(layers.Dense(128,activation=('relu')))
        #model_1.add(Dropout(.2))
        network.add(layers.Dense(10,activation=('softmax')))


        default_transformer_kwargs = {
            "network": network,
            "euclidean_layer_idx": -1,
            "loss": "categorical_crossentropy",
            "optimizer": Adam(3e-4),
            "fit_kwargs": {
                "epochs": 20,
                #"validation_split": 0.33,
                "verbose": True,
                "batch_size": 32,
                "shuffle": True
            },
        }
        default_voter_class = KNNClassificationVoter
        default_voter_kwargs = {"k" : int(np.log2(num_points_per_task))}

        default_decider_class = SimpleArgmaxAverage


    progressive_learner = ProgressiveLearner(default_transformer_class = default_transformer_class,
                                         default_transformer_kwargs = default_transformer_kwargs,
                                         default_voter_class = default_voter_class,
                                         default_voter_kwargs = default_voter_kwargs,
                                         default_decider_class = default_decider_class)

    for task_ii in range(10):
        print("Starting Task {} For Fold {}".format(task_ii, shift))


        train_start_time = time.time()
        
        if acorn is not None:
            np.random.seed(acorn)

        progressive_learner.add_task(
            X = train_x[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task],
            y = train_y[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task],
            num_transformers = 1 if model == "dnn" else ntrees,
            transformer_voter_decider_split = [0.63, 0.37, 0],
            decider_kwargs = {"classes" : np.unique(train_y[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task])}
            )
        train_end_time = time.time()
        
        '''single_learner = ProgressiveLearner(default_transformer_class = default_transformer_class,
                                         default_transformer_kwargs = default_transformer_kwargs,
                                         default_voter_class = default_voter_class,
                                         default_voter_kwargs = default_voter_kwargs,
                                         default_decider_class = default_decider_class)

        if acorn is not None:
            np.random.seed(acorn)

        single_learner.add_task(
            X = train_x[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task],
            y = train_y[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task],
            num_transformers = 1 if model == "dnn" else (task_ii+1)*ntrees,
            transformer_voter_decider_split = [0.67, 0.33, 0],
            decider_kwargs = {"classes" : np.unique(train_y[task_ii*5000+slot*num_points_per_task:task_ii*5000+(slot+1)*num_points_per_task])}
            )'''

        train_times_across_tasks.append(train_end_time - train_start_time)

        single_task_inference_start_time = time.time()
        single_task=progressive_learner.predict(
            X = test_x[task_ii*1000:(task_ii+1)*1000,:], transformer_ids=[task_ii], task_id=task_ii
            )
        single_task_inference_end_time = time.time()
        single_task_accuracies[task_ii] = np.mean(
                single_task == test_y[task_ii*1000:(task_ii+1)*1000]
                    )
        single_task_inference_times_across_tasks.append(single_task_inference_end_time - single_task_inference_start_time)



        for task_jj in range(task_ii+1):
            multitask_inference_start_time = time.time()
            llf_task=progressive_learner.predict(
                X = test_x[task_jj*1000:(task_jj+1)*1000,:], task_id=task_jj
                )
            multitask_inference_end_time = time.time()

            shifts.append(shift)
            tasks.append(task_jj+1)
            base_tasks.append(task_ii+1)
            accuracies_across_tasks.append(np.mean(
                llf_task == test_y[task_jj*1000:(task_jj+1)*1000]
                ))
            multitask_inference_times_across_tasks.append(multitask_inference_end_time - multitask_inference_start_time)

    df['data_fold'] = shifts
    df['task'] = tasks
    df['base_task'] = base_tasks
    df['accuracy'] = accuracies_across_tasks
    df_single_task = pd.DataFrame()
    df_single_task['task'] = range(1, 11)
    df_single_task['data_fold'] = shift
    df_single_task['accuracy'] = single_task_accuracies
 
    print(df)
    summary = (df,df_single_task)
    file_to_save = '/Users/jayantadey/ProgLearn/benchmarks/cifar_exp/result_pretrained/'+model+'_not_pretrained_'+ str(slot)+'_'+str(shift)+'.pickle'
    with open(file_to_save, 'wb') as f:
        pickle.dump(summary, f)

    '''file_to_save = 'result/time_res/'+model+str(ntrees)+'_'+str(shift)+'_'+str(slot)+'.pickle'
    with open(file_to_save, 'wb') as f:
        pickle.dump(time_info, f)

    file_to_save = 'result/mem_res/'+model+str(ntrees)+'_'+str(shift)+'_'+str(slot)+'.pickle'
    with open(file_to_save, 'wb') as f:
        pickle.dump(mem_info, f)'''

#%%
def cross_val_data(data_x, data_y, num_points_per_task, total_task=10, shift=1):
    x = data_x.copy()#.astype('int')
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
    print('conversion done!')
    return train_x, train_y, test_x, test_y

#%%
def run_parallel_exp(data_x, data_y, n_trees, model, num_points_per_task, slot=0, shift=1):
    train_x, train_y, test_x, test_y = cross_val_data(data_x, data_y, num_points_per_task, shift=shift)
    LF_experiment(train_x, train_y, test_x, test_y, n_trees, shift, slot, model, num_points_per_task, acorn=12345)

#%%
### MAIN HYPERPARAMS ###
model = "dnn"
num_points_per_task = 500
########################

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]


#%%
slot_fold = range(10)
shift_fold = range(6)

iterable = product(shift_fold,slot_fold)

for shift,slot in iterable:
    run_parallel_exp(data_x, data_y, 0, model, num_points_per_task, slot=slot, shift=shift)



#%%
# train_x, train_y, test_x, test_y = cross_val_data(data_x, data_y, num_points_per_task, shift=shift)

# X = train_x[0*5000+1*500:0*5000+(1+1)*500]
# y = to_categorical(train_y[0*5000+1*500:0*5000+(1+1)*500])
# x_test = test_x[0*1000:(0+1)*1000,:]
# y_test = to_categorical(test_y[0*1000:(0+1)*1000])
# #%%
# model = keras.Sequential()
# base_model_1 = ResNet50(include_top=False,weights=None,input_shape=(32,32,3),classes=y_train.shape[1])
# model.add(base_model_1) #Adds the base model (in this case vgg19 to model_1)
# model.add(layers.Flatten())
# model.add(layers.Dense(1024,activation=('relu'),input_dim=512))
# model.add(layers.Dense(512,activation=('relu'))) 
# model.add(layers.Dense(256,activation=('relu'))) 
# #model_1.add(Dropout(.3))#Adding a dropout layer that will randomly drop 30% of the weights
# model.add(layers.Dense(128,activation=('relu')))
# #model_1.add(Dropout(.2))
# model.add(layers.Dense(10,activation=('softmax'))) #This is the classification layer
# # %%
# model.compile(loss='categorical_crossentropy',
#             optimizer=Adam(lr=lr_schedule(0)),
#             metrics=['accuracy'])
# # %%
# model.fit(X, y,
#         batch_size=32,
#         epochs=20,
#         validation_data=(x_test, y_test),
#         shuffle=True)
# # %%
