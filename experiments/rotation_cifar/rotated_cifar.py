import matplotlib.pyplot as plt
import random
import pickle
from skimage.transform import rotate
from scipy import ndimage
from skimage.util import img_as_ubyte
from joblib import Parallel, delayed
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.ensemble.forest import _generate_sample_indices
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from itertools import product
import keras

from keras import layers

from joblib import Parallel, delayed
from multiprocessing import Pool

import tensorflow as tf

from numba import cuda

import sys
sys.path.append("../../proglearn/")
from progressive_learner import ProgressiveLearner
from deciders import SimpleAverage
from transformers import TreeClassificationTransformer, NeuralClassificationTransformer 
from voters import TreeClassificationVoter, KNNClassificationVoter

def cross_val_data(data_x, data_y, total_cls=10):
    x = data_x.copy()
    y = data_y.copy()
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]
    
    
    for i in range(total_cls):
        indx = idx[i]#np.roll(idx[i],(cv-1)*100)
        random.shuffle(indx)
        
        if i==0:
            train_x1 = x[indx[0:250],:]
            train_x2 = x[indx[250:500],:]
            train_y1 = y[indx[0:250]]
            train_y2 = y[indx[250:500]]
            
            test_x = x[indx[500:600],:]
            test_y = y[indx[500:600]]
        else:
            train_x1 = np.concatenate((train_x1, x[indx[0:250],:]), axis=0)
            train_x2 = np.concatenate((train_x2, x[indx[250:500],:]), axis=0)
            train_y1 = np.concatenate((train_y1, y[indx[0:250]]), axis=0)
            train_y2 = np.concatenate((train_y2, y[indx[250:500]]), axis=0)
            
            test_x = np.concatenate((test_x, x[indx[500:600],:]), axis=0)
            test_y = np.concatenate((test_y, y[indx[500:600]]), axis=0)
        
        
    return train_x1, train_y1, train_x2, train_y2, test_x, test_y 

def LF_experiment(data_x, data_y, angle, model, granularity, reps=1, ntrees=29, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
    
    errors = np.zeros(2)
    
    for rep in range(reps):
        print("Starting Rep {} of Angle {}".format(rep, angle))
        train_x1, train_y1, train_x2, train_y2, test_x, test_y = cross_val_data(data_x, data_y, total_cls=10)
    
    
        #change data angle for second task
        tmp_data = train_x2.copy()
        _tmp_ = np.zeros((32,32,3), dtype=int)
        total_data = tmp_data.shape[0]
        
        for i in range(total_data):
            tmp_ = image_aug(tmp_data[i],angle)
            tmp_data[i] = tmp_
        
        if model == "uf":
            train_x1 = train_x1.reshape((train_x1.shape[0], train_x1.shape[1] * train_x1.shape[2] * train_x1.shape[3]))
            tmp_data = tmp_data.reshape((tmp_data.shape[0], tmp_data.shape[1] * tmp_data.shape[2] * tmp_data.shape[3]))
            test_x = test_x.reshape((test_x.shape[0], test_x.shape[1] * test_x.shape[2] * test_x.shape[3]))
            
        if model == "dnn":
            with tf.device('/gpu:'+str(int(angle //  granularity) % 4)):
                default_transformer_class = NeuralClassificationTransformer

                network = keras.Sequential()
                network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=np.shape(train_x1)[1:]))
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
                default_voter_kwargs = {"k" : int(np.log2(len(train_x1)))}

                default_decider_class = SimpleAverage
        elif model == "uf":
            default_transformer_class = TreeClassificationTransformer
            default_transformer_kwargs = {"kwargs" : {"max_depth" : 30}}

            default_voter_class = TreeClassificationVoter
            default_voter_kwargs = {}

            default_decider_class = SimpleAverage
        
            
        progressive_learner = ProgressiveLearner(default_transformer_class = default_transformer_class, 
                                     default_transformer_kwargs = default_transformer_kwargs,
                                     default_voter_class = default_voter_class,
                                     default_voter_kwargs = default_voter_kwargs,
                                     default_decider_class = default_decider_class)

        progressive_learner.add_task(
            X = train_x1, 
            y = train_y1,
            transformer_voter_decider_split = [0.67, 0.33, 0],
            decider_kwargs = {"classes" : np.unique(train_y1)}
        )

        progressive_learner.add_transformer(
            X = tmp_data, 
            y = train_y2,
            transformer_data_proportion = 1,
            backward_task_ids = [0]
        )


        llf_task1=progressive_learner.predict(test_x, task_id=0)
        llf_single_task=progressive_learner.predict(test_x, task_id=0, transformer_ids=[0])

        errors[1] = errors[1]+(1 - np.mean(llf_task1 == test_y))
        errors[0] = errors[0]+(1 - np.mean(llf_single_task == test_y))
    
    errors = errors/reps
    print("Errors For Angle {}: {}".format(angle, errors))
    with open('results/angle_'+str(angle)+'_'+model+'.pickle', 'wb') as f:
        pickle.dump(errors, f, protocol = 2)
        
def image_aug(pic, angle, centroid_x=23, centroid_y=23, win=16, scale=1.45):
    im_sz = int(np.floor(pic.shape[0]*scale))
    pic_ = np.uint8(np.zeros((im_sz,im_sz,3),dtype=int))
    
    pic_[:,:,0] = ndimage.zoom(pic[:,:,0],scale)
    
    pic_[:,:,1] = ndimage.zoom(pic[:,:,1],scale)
    pic_[:,:,2] = ndimage.zoom(pic[:,:,2],scale)
    
    image_aug = rotate(pic_, angle, resize=False)
    #print(image_aug.shape)
    image_aug_ = image_aug[centroid_x-win:centroid_x+win,centroid_y-win:centroid_y+win,:]
    
    return img_as_ubyte(image_aug_)

### MAIN HYPERPARAMS ###
model = "uf"
granularity = 2
reps = 4
########################


(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]

def perform_angle(angle):
    LF_experiment(data_x, data_y, angle, model, granularity, reps=reps, ntrees=16, acorn=1)

if model == "dnn":
    for angle_adder in range(0, 180, granularity * 4):
        angles = angle_adder + np.arange(0, granularity * 4, granularity)
        with Pool(4) as p:
            p.map(perform_angle, angles)
elif model == "uf":
    angles = np.arange(0, 180,2)
    with Pool(96) as p:
        p.map(perform_angle, angles)