import random
import tensorflow as tf
import numpy as np
import pickle
from joblib import Parallel, delayed
from math import log2, ceil
from proglearn.forest import LifelongClassificationForest
from proglearn.sims import generate_spirals


def experiment(n_task1, n_task2, n_test=1000, 
               n_trees=10, max_depth=None, random_state=None):
    
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


    progressive_learner = LifelongClassificationForest(default_n_estimators=n_trees)
    uf1 = LifelongClassificationForest(default_n_estimators=n_trees)
    naive_uf = LifelongClassificationForest(default_n_estimators=n_trees)
    uf2 = LifelongClassificationForest(default_n_estimators=n_trees)
    
    #source data
    X_task1, y_task1 = generate_spirals(n_task1, 3, noise=0.8)
    test_task1, test_label_task1 = generate_spirals(n_test,  3, noise=0.8)

    #target data
    X_task2, y_task2 = generate_spirals(n_task2,  5, noise=0.4)
    test_task2, test_label_task2 = generate_spirals(n_test,  5, noise=0.4)

    if n_task1 == 0:
        progressive_learner.add_task(X_task2, y_task2, n_estimators=n_trees)
        uf2.add_task(X_task2, y_task2, n_estimators=n_trees)

        errors[0] = 0.5
        errors[1] = 0.5

        uf_task2=uf2.predict(test_task2, task_id=0)
        l2f_task2=progressive_learner.predict(test_task2, task_id=0)

        errors[2] = 1 - np.mean(uf_task2 == test_label_task2)
        errors[3] = 1 - np.mean(l2f_task2 == test_label_task2)
        
        errors[4] = 0.5
        errors[5] = 1 - np.mean(uf_task2 == test_label_task2)
    elif n_task2 == 0:
        progressive_learner.add_task(X_task1, y_task1,
                                     n_estimators=n_trees)
        uf1.add_task(X_task1, y_task1,
                                     n_estimators=n_trees)

        uf_task1=uf1.predict(test_task1, task_id=0)
        l2f_task1=progressive_learner.predict(test_task1, task_id=0)

        errors[0] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[1] = 1 - np.mean(l2f_task1 == test_label_task1)
        
        errors[2] = 0.5
        errors[3] = 0.5
        
        errors[4] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[5] = 0.5
    else:
        progressive_learner.add_task(X_task1, y_task1, n_estimators=n_trees)
        progressive_learner.add_task(X_task2, y_task2, n_estimators=n_trees)

        uf1.add_task(X_task1, y_task1, n_estimators=2*n_trees)
        uf2.add_task(X_task2, y_task2, n_estimators=2*n_trees)
        
        naive_uf_train_x = np.concatenate((X_task1,X_task2),axis=0)
        naive_uf_train_y = np.concatenate((y_task1,y_task2),axis=0)
        naive_uf.add_task(
                naive_uf_train_x, naive_uf_train_y, n_estimators=n_trees
                )
        
        uf_task1=uf1.predict(test_task1, task_id=0)
        l2f_task1=progressive_learner.predict(test_task1, task_id=0)
        uf_task2=uf2.predict(test_task2, task_id=0)
        l2f_task2=progressive_learner.predict(test_task2, task_id=1)
        naive_uf_task1 = naive_uf.predict(
            test_task1, task_id=0
        )
        naive_uf_task2 = naive_uf.predict(
            test_task2, task_id=0
        )

        errors[0] = 1 - np.mean(
            uf_task1 == test_label_task1
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
            naive_uf_task1 == test_label_task1
        )
        errors[5] = 1 - np.mean(
            naive_uf_task2 == test_label_task2
        )

    return errors



mc_rep = 100
n_test = 1000
n_trees = 10
n_spiral3 = (100*np.arange(0.5, 7.75, step=0.25)).astype(int)
n_spiral5 = (100*np.arange(0.25, 7.75, step=0.25)).astype(int)

mean_error = np.zeros((6, len(n_spiral3)+len(n_spiral5)))
std_error = np.zeros((6, len(n_spiral3)+len(n_spiral5)))

mean_te = np.zeros((4, len(n_spiral3)+len(n_spiral5)))
std_te = np.zeros((4, len(n_spiral3)+len(n_spiral5)))

for i,n1 in enumerate(n_spiral3):
    print('starting to compute %s for 3 spirals\n'%n1)
    error = np.array(
        Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment)(
            n1,0,max_depth=ceil(log2(n1))
        ) for _ in range(mc_rep)
      )
    )
    mean_error[:,i] = np.mean(error,axis=0)
    std_error[:,i] = np.std(error,ddof=1,axis=0)
    mean_te[0,i] = np.mean(error[:,0])/np.mean(error[:,1])
    mean_te[1,i] = np.mean(error[:,2])/np.mean(error[:,3])
    mean_te[2,i] = np.mean(error[:,0])/np.mean(error[:,4])
    mean_te[3,i] = np.mean(error[:,2])/np.mean(error[:,5])
    

    if n1==n_spiral3[-1]:
        for j,n2 in enumerate(n_spiral5):
            print('starting to compute %s for 5 spirals\n'%n2)
            
            error = np.array(
                Parallel(n_jobs=-1,verbose=1)(
                delayed(experiment)(
                    n1,n2,max_depth=ceil(log2(750))
                ) for _ in range(mc_rep)
              )
            )
            mean_error[:,i+j+1] = np.mean(error,axis=0)
            std_error[:,i+j+1] = np.std(error,ddof=1,axis=0)
            mean_te[0,i+j+1] = np.mean(error[:,0])/np.mean(error[:,1])
            mean_te[1,i+j+1] = np.mean(error[:,2])/np.mean(error[:,3])
            mean_te[2,i+j+1] = np.mean(error[:,0])/np.mean(error[:,4])
            mean_te[3,i+j+1] = np.mean(error[:,2])/np.mean(error[:,5])

with open('mean_spiral.pickle','wb') as f:
    pickle.dump(mean_error,f)
    
with open('std_spiral.pickle','wb') as f:
    pickle.dump(std_error,f)
    
with open('mean_te_spiral.pickle','wb') as f:
    pickle.dump(mean_te,f)
    