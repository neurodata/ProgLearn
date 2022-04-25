#%%
import random
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import pickle
from joblib import Parallel, delayed
from math import log2, ceil
from proglearn.progressive_learner import ProgressiveLearner
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import NeuralClassificationTransformer, TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter
from proglearn.sims import generate_gaussian_parity
from keras.utils.np_utils import to_categorical


def experiment(n_task1, n_task2, n_test=1000, 
               task1_angle=0, task2_angle=np.pi/2, 
               n_trees=1, max_depth=None, random_state=None):
    
    """
    A function to do progressive experiment between two tasks
    where the task data is generated using Gaussian parity.
    
    Parameters
    ----------
    n_task1 : int
        Total number of train sample for task 1.
    
    n_task2 : int
        Total number of train dsample for task 2

    n_test : int, optional (default=1000)
        Number of test sample for each task.
        
    task1_angle : float, optional (default=0)
        Angle in radian for task 1.
            
    task2_angle : float, optional (default=numpy.pi/2)
        Angle in radian for task 2.
            
    n_trees : int, optional (default=10)
        Number of total trees to train for each task.

    max_depth : int, optional (default=None)
        Maximum allowable depth for each tree.
        
    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        
    
    Returns
    -------
    errors : array of shape [6]
        Elements of the array is organized as single task error task1,
        multitask error task1, single task error task2,
        multitask error task2, naive UF error task1,
        naive UF task2.
    """

    if n_task1==0 and n_task2==0:
        raise ValueError('Wake up and provide samples to train!!!')

    if random_state != None:
        np.random.seed(random_state)

    errors = np.zeros(6,dtype=float)

    default_transformer_class = NeuralClassificationTransformer
    network = keras.Sequential()
    network.add(keras.Input(shape=(2,)))
    network.add(layers.Dense(2, activation='relu'))
    network.add(layers.Dense(10, activation='relu'))
    network.add(layers.Dense(10, activation='relu'))
    network.add(layers.Dense(units=2, activation = 'softmax'))

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
    default_voter_kwargs = {"k" : int(np.log2(750))}

    default_decider_class = SimpleArgmaxAverage
    default_decider_kwargs = {"classes" : np.arange(2)}
        
    '''default_transformer_class = TreeClassificationTransformer
    default_transformer_kwargs = {"kwargs" : {"max_depth" : max_depth}}

    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = {}

    default_decider_class = SimpleArgmaxAverage
    default_decider_kwargs = {"classes" : np.arange(2)}'''
    progressive_learner = ProgressiveLearner(default_transformer_class = default_transformer_class,
                                            default_transformer_kwargs = default_transformer_kwargs,
                                            default_voter_class = default_voter_class,
                                            default_voter_kwargs = default_voter_kwargs,
                                            default_decider_class = default_decider_class,
                                            default_decider_kwargs = default_decider_kwargs)
    nn = ProgressiveLearner(default_transformer_class = default_transformer_class,
                                            default_transformer_kwargs = default_transformer_kwargs,
                                            default_voter_class = default_voter_class,
                                            default_voter_kwargs = default_voter_kwargs,
                                            default_decider_class = default_decider_class,
                                            default_decider_kwargs = default_decider_kwargs)
    naive_nn = ProgressiveLearner(default_transformer_class = default_transformer_class,
                                            default_transformer_kwargs = default_transformer_kwargs,
                                            default_voter_class = default_voter_class,
                                            default_voter_kwargs = default_voter_kwargs,
                                            default_decider_class = default_decider_class,
                                            default_decider_kwargs = default_decider_kwargs)
    
    #source data
    X_task1, y_task1 = generate_gaussian_parity(n_task1, angle_params=task1_angle)
    test_task1, test_label_task1 = generate_gaussian_parity(n_test, angle_params=task1_angle)

    #target data
    X_task2, y_task2 = generate_gaussian_parity(n_task2, angle_params=task2_angle)
    test_task2, test_label_task2 = generate_gaussian_parity(n_test, angle_params=task2_angle)

    if n_task1 == 0:
        progressive_learner.add_task(X_task2, y_task2, num_transformers=n_trees)

        errors[0] = 0.5
        errors[1] = 0.5

        uf_task2=progressive_learner.predict(test_task2,
                                             transformer_ids=[0], task_id=0)
        l2f_task2=progressive_learner.predict(test_task2, task_id=0)

        errors[2] = 1 - np.mean(uf_task2 == test_label_task2)
        errors[3] = 1 - np.mean(l2f_task2 == test_label_task2)
        
        errors[4] = 0.5
        errors[5] = 1 - np.mean(uf_task2 == test_label_task2)
    elif n_task2 == 0:
        progressive_learner.add_task(X_task1, y_task1,
                                     num_transformers=n_trees)

        nn_task1=progressive_learner.predict(test_task1, 
                                             transformer_ids=[0], task_id=0)
        l2f_task1=progressive_learner.predict(test_task1, task_id=0)

        errors[0] = 1 - np.mean(nn_task1 == test_label_task1)
        errors[1] = 1 - np.mean(l2f_task1 == test_label_task1)
        
        errors[2] = 0.5
        errors[3] = 0.5
        
        errors[4] = 1 - np.mean(nn_task1 == test_label_task1)
        errors[5] = 0.5
    else:
        progressive_learner.add_task(X_task1, y_task1, num_transformers=n_trees)
        progressive_learner.add_task(X_task2, y_task2, num_transformers=n_trees)

        nn.add_task(X_task1, y_task1, num_transformers=2*n_trees)
        nn.add_task(X_task2, y_task2, num_transformers=2*n_trees)
        
        naive_nn_train_x = np.concatenate((X_task1,X_task2),axis=0)
        naive_nn_train_y = np.concatenate((y_task1,y_task2),axis=0)
        naive_nn.add_task(
                naive_nn_train_x, naive_nn_train_y, num_transformers=n_trees
                )
        
        nn_task1=nn.predict(test_task1, transformer_ids=[0], task_id=0)
        l2f_task1=progressive_learner.predict(test_task1, task_id=0)
        uf_task2=nn.predict(test_task2, transformer_ids=[1], task_id=1)
        l2f_task2=progressive_learner.predict(test_task2, task_id=1)
        naive_nn_task1 = naive_nn.predict(
            test_task1, transformer_ids=[0], task_id=0
        )
        naive_nn_task2 = naive_nn.predict(
            test_task2, transformer_ids=[0], task_id=0
        )

        errors[0] = 1 - np.mean(
            nn_task1 == test_label_task1
        )
        errors[1] = 1 - np.mean(
            l2f_task1 == test_label_task1
        )
        errors[2] = 1 - np.mean(
            uf_task2 == test_label_task2
        )
        errors[3] = 1 - np.mean(
            l2f_task2 == test_label_task2
        )
        errors[4] = 1 - np.mean(
            naive_nn_task1 == test_label_task1
        )
        errors[5] = 1 - np.mean(
            naive_nn_task2 == test_label_task2
        )

    return errors

#%%
mc_rep = 1000
n_test = 1000
n_trees = 1
n_xor = (100*np.arange(0.5, 7.50, step=0.25)).astype(int)
n_nxor = (100*np.arange(0.25, 7.50, step=0.25)).astype(int)

error = np.zeros((mc_rep,6), dtype=float)
mean_error = np.zeros((6, len(n_xor)+len(n_nxor)))
std_error = np.zeros((6, len(n_xor)+len(n_nxor)))

mean_te = np.zeros((4, len(n_xor)+len(n_nxor)))
std_te = np.zeros((4, len(n_xor)+len(n_nxor)))

for i,n1 in enumerate(n_xor):
    print('starting to compute %s xor\n'%n1)

    for ii in range(mc_rep):
        error[ii] = experiment(
            n1,0,max_depth=ceil(log2(n1))
        ) 
    mean_error[:,i] = np.mean(error,axis=0)
    std_error[:,i] = np.std(error,ddof=1,axis=0)
    mean_te[0,i] = np.mean(error[:,0])/np.mean(error[:,1])
    mean_te[1,i] = np.mean(error[:,2])/np.mean(error[:,3])
    mean_te[2,i] = np.mean(error[:,0])/np.mean(error[:,4])
    mean_te[3,i] = np.mean(error[:,2])/np.mean(error[:,5])
    

    if n1==n_xor[-1]:
        for j,n2 in enumerate(n_nxor):
            print('starting to compute %s nxor\n'%n2)
            
            for jj in range(mc_rep):
                error[jj] = experiment(
                    n1,n2,max_depth=ceil(log2(750))
                ) 
              
            mean_error[:,i+j+1] = np.mean(error,axis=0)
            std_error[:,i+j+1] = np.std(error,ddof=1,axis=0)
            mean_te[0,i+j+1] = np.mean(error[:,0])/np.mean(error[:,1])
            mean_te[1,i+j+1] = np.mean(error[:,2])/np.mean(error[:,3])
            mean_te[2,i+j+1] = np.mean(error[:,0])/np.mean(error[:,4])
            mean_te[3,i+j+1] = np.mean(error[:,2])/np.mean(error[:,5])

with open('./data/mean_xor_nxor_nn.pickle','wb') as f:
    pickle.dump(mean_error,f)
    
with open('./data/std_xor_nxor_nn.pickle','wb') as f:
    pickle.dump(std_error,f)
    
with open('./data/mean_te_xor_nxor_nn.pickle','wb') as f:
    pickle.dump(mean_te,f)
    



#%%
mc_rep = 1000
n_test = 1000
n_trees = 1
n_xor = (100*np.arange(0.5, 7.50, step=0.25)).astype(int)
n_rxor = (100*np.arange(0.25, 7.50, step=0.25)).astype(int)

error = np.zeros((mc_rep,6), dtype=float)
mean_error = np.zeros((6, len(n_xor)+len(n_rxor)))
std_error = np.zeros((6, len(n_xor)+len(n_rxor)))

mean_te = np.zeros((4, len(n_xor)+len(n_rxor)))
std_te = np.zeros((4, len(n_xor)+len(n_rxor)))

for i,n1 in enumerate(n_xor):
    print('starting to compute %s xor\n'%n1)

    for ii in range(mc_rep):
        error[ii] = experiment(
                n1,0,task2_angle=np.pi/4,
                max_depth=ceil(log2(750))
            )
    
    mean_error[:,i] = np.mean(error,axis=0)
    std_error[:,i] = np.std(error,ddof=1,axis=0)
    mean_te[0,i] = np.mean(error[:,0])/np.mean(error[:,1])
    mean_te[1,i] = np.mean(error[:,2])/np.mean(error[:,3])
    mean_te[2,i] = np.mean(error[:,0])/np.mean(error[:,4])
    mean_te[3,i] = np.mean(error[:,2])/np.mean(error[:,5])

    if n1==n_xor[-1]:
        for j,n2 in enumerate(n_rxor):
            print('starting to compute %s rxor\n'%n2)
            
            for jj in range(mc_rep):
                error[jj] = experiment(
                    n1,n2,task2_angle=np.pi/4,
                    max_depth=ceil(log2(750))
                )

            mean_error[:,i+j+1] = np.mean(error,axis=0)
            std_error[:,i+j+1] = np.std(error,ddof=1,axis=0)
            mean_te[0,i+j+1] = np.mean(error[:,0])/np.mean(error[:,1])
            mean_te[1,i+j+1] = np.mean(error[:,2])/np.mean(error[:,3])
            mean_te[2,i+j+1] = np.mean(error[:,0])/np.mean(error[:,4])
            mean_te[3,i+j+1] = np.mean(error[:,2])/np.mean(error[:,5])

with open('./data/mean_xor_rxor_nn.pickle','wb') as f:
    pickle.dump(mean_error,f)
    
with open('./data/std_xor_rxor_nn.pickle','wb') as f:
    pickle.dump(std_error,f)
    
with open('./data/mean_te_xor_rxor_nn.pickle','wb') as f:
    pickle.dump(mean_te,f)



#%%
###main hyperparameters###
angle_sweep = range(0,91,1)
task1_sample = 100
task2_sample = 100
mc_rep = 10000

error = np.zeros((mc_rep,6), dtype=float)
mean_te = np.zeros(len(angle_sweep), dtype=float)
for ii,angle in enumerate(angle_sweep):
    print('angle '+str(angle))
    for jj in range(mc_rep):
        error[jj] = experiment(
                task1_sample,task2_sample,
                task2_angle=angle*np.pi/180, 
                max_depth=ceil(log2(task1_sample))
            ) 
    mean_te[ii] = np.mean(error[:,0])/np.mean(error[:,1])

with open('./data/mean_angle_te_nn.pickle','wb') as f:
    pickle.dump(mean_te,f)

#%%
###main hyperparameters###
task2_sample_sweep = (2**np.arange(np.log2(60), np.log2(5010)+1, .25)).astype('int')
task1_sample = 500
task2_angle = 25*np.pi/180
mc_rep = 1000

error = np.zeros((mc_rep,6), dtype=float)
mean_te = np.zeros(len(task2_sample_sweep), dtype=float)
for ii,sample_no in enumerate(task2_sample_sweep):

    for jj in range(mc_rep):
        error[jj] = experiment(
                task1_sample,sample_no,
                task2_angle=task2_angle, 
                max_depth=None
            )

    mean_te[ii] = np.mean(error[:,0])/np.mean(error[:,1])

with open('./data/mean_sample_te_nn.pickle','wb') as f:
    pickle.dump(mean_te,f)