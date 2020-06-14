#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from itertools import product
import pandas as pd

import numpy as np
import pickle
from math import log2, ceil 

import sys
sys.path.append("../../src/")
from lifelong_dnn import LifeLongDNN
from joblib import Parallel, delayed

import tensorflow as tf

#%%
############################
### Main hyperparameters ###
############################
ntrees = 50
hybrid_comp_trees = 25
estimation_set = 0.63
validation_set= 1-estimation_set
num_points_per_task = 5000
reps = 20
task_10_sample = range(500,num_points_per_task+500,500)

#%%
def sort_data(data_x, data_y, num_points_per_task, total_task=10, shift=1):
    x = data_x.copy()
    y = data_y.copy()
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]
    train_x_across_task = []
    train_y_across_task = []
    test_x_across_task = []
    test_y_across_task = []

    batch_per_task=5000//num_points_per_task
    sample_per_class = num_points_per_task//total_task
    test_data_slot=100//batch_per_task

    for task in range(total_task):
        for batch in range(batch_per_task):
            for class_no in range(task*10,(task+1)*10,1):
                indx = np.roll(idx[class_no],(shift-1)*100)
                
                if batch==0 and class_no==task*10:
                    train_x = x[indx[batch*sample_per_class:(batch+1)*sample_per_class],:]
                    train_y = y[indx[batch*sample_per_class:(batch+1)*sample_per_class]]
                    test_x = x[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500],:]
                    test_y = y[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500]]
                else:
                    train_x = np.concatenate((train_x, x[indx[batch*sample_per_class:(batch+1)*sample_per_class],:]), axis=0)
                    train_y = np.concatenate((train_y, y[indx[batch*sample_per_class:(batch+1)*sample_per_class]]), axis=0)
                    test_x = np.concatenate((test_x, x[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500],:]), axis=0)
                    test_y = np.concatenate((test_y, y[indx[batch*test_data_slot+500:(batch+1)*test_data_slot+500]]), axis=0)
        
        train_x_across_task.append(train_x)
        train_y_across_task.append(train_y)
        test_x_across_task.append(test_x)
        test_y_across_task.append(test_y)

    return train_x_across_task, train_y_across_task, test_x_across_task, test_y_across_task

# %%
def voter_predict_proba(voter, nodes_across_trees):
            def worker(tree_idx):
                #get the node_ids_to_posterior_map for this tree
                node_ids_to_posterior_map = voter.tree_idx_to_node_ids_to_posterior_map[tree_idx]

                #get the nodes of X
                nodes = nodes_across_trees[tree_idx]

                posteriors = []
                node_ids = node_ids_to_posterior_map.keys()

                #loop over nodes of X
                for node in nodes:
                    #if we've seen this node before, simply get the posterior
                    if node in node_ids:
                        posteriors.append(node_ids_to_posterior_map[node])
                    #if we haven't seen this node before, simply use the uniform posterior 
                    else:
                        posteriors.append(np.ones((len(np.unique(voter.classes_)))) / len(voter.classes_))
                return posteriors

            if voter.parallel:
                return Parallel(n_jobs=-1)(
                                delayed(worker)(tree_idx) for tree_idx in range(voter.n_estimators)
                        )

            else:
                return [worker(tree_idx) for tree_idx in range(voter.n_estimators)]

#%%
def estimate_posteriors(l2f, X, representation = 0, decider = 0):
        l2f.check_task_idx_(decider)
        
        if representation == "all":
            representation = range(l2f.n_tasks)
        elif isinstance(representation, int):
            representation = np.array([representation])
        
        def worker(transformer_task_idx):
            transformer = l2f.transformers_across_tasks[transformer_task_idx]
            voter = l2f.voters_across_tasks_matrix[decider][transformer_task_idx]

            return voter_predict_proba(voter,transformer(X))
        
        '''if l2f.parallel:
            posteriors_across_tasks = np.array(
                        Parallel(n_jobs=-1)(
                                delayed(worker)(transformer_task_idx) for transformer_task_idx in representation
                        )
                )    
        else:'''
        posteriors_across_tasks = np.array([worker(transformer_task_idx) for transformer_task_idx in representation])    

        return posteriors_across_tasks

# %%
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
data_x = data_x.reshape((data_x.shape[0], data_x.shape[1] * data_x.shape[2] * data_x.shape[3]))
data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]

train_x_across_task, train_y_across_task, test_x_across_task, test_y_across_task = sort_data(
    data_x,data_y,num_points_per_task
    )
# %%
hybrid = np.zeros(reps,dtype=float)
building = np.zeros(reps,dtype=float)
recruiting= np.zeros(reps,dtype=float)
uf = np.zeros(reps,dtype=float)
mean_accuracy_dict = {'hybrid':[],'bulding':[],'recruiting':[],'UF':[]}
std_accuracy_dict = {'hybrid':[],'bulding':[],'recruiting':[],'UF':[]}

for ns in task_10_sample:
    estimation_sample_no = ceil(estimation_set*ns)
    validation_sample_no = ns - estimation_sample_no

    for rep in range(reps):
        print("doing {} samples for {} th rep".format(ns,rep))
        ## estimation
        l2f = LifeLongDNN(model = "uf", parallel = True)

        for task in range(9):
            l2f.new_forest(
            train_x_across_task[task], 
            train_y_across_task[task], 
            max_depth=ceil(log2(num_points_per_task)),
            n_estimators=ntrees
            )

        task_10_train_indx = np.random.choice(num_points_per_task, ns, replace=False)

        l2f.new_forest(
            train_x_across_task[9][task_10_train_indx[:estimation_sample_no]], 
            train_y_across_task[9][task_10_train_indx[:estimation_sample_no]], 
            max_depth=ceil(log2(estimation_sample_no)),
            n_estimators=ntrees
            )

        l2f_ = LifeLongDNN(model = "uf", parallel = True)
        l2f_.new_forest(
            train_x_across_task[9][task_10_train_indx], 
            train_y_across_task[9][task_10_train_indx], 
            max_depth=ceil(log2(ns)),
            n_estimators=hybrid_comp_trees
            )

        
        ## validation
        posteriors_across_trees = estimate_posteriors(
            l2f,
            train_x_across_task[9][task_10_train_indx[estimation_sample_no:]],
            representation=[0,1,2,3,4,5,6,7,8],
            decider=9
            )
        
        posteriors_across_trees = posteriors_across_trees.reshape(
            9*ntrees,
            validation_sample_no,
            10
            )

        error_across_trees = np.zeros(9*ntrees)
        validation_target = train_y_across_task[9][task_10_train_indx[estimation_sample_no:]]
        for tree in range(9*ntrees):
            res = np.argmax(posteriors_across_trees[tree],axis=1) + 90
            error_across_trees[tree] = 1-np.mean(
                validation_target==res
            )

        best_50_tree = np.argsort(error_across_trees)[:50]
        best_25_tree = np.argsort(error_across_trees)[:25]

        ## evaluation
        posteriors_across_trees = estimate_posteriors(
            l2f,
            test_x_across_task[9],
            representation=[0,1,2,3,4,5,6,7,8],
            decider=9
            )
        posteriors_across_trees = posteriors_across_trees.reshape(
            9*ntrees,
            1000,
            10
            )

        recruiting_posterior = np.mean(posteriors_across_trees[best_50_tree],axis=0)
        res = np.argmax(recruiting_posterior,axis=1) + 90
        recruiting[rep] = 1 - np.mean(
                test_y_across_task[9]==res
            )

        building_res = l2f.predict(
            test_x_across_task[9],
            representation=[0,1,2,3,4,5,6,7,8,9],
            decider=9
        )
        building[rep] = 1 - np.mean(
                test_y_across_task[9]==building_res
            )

        uf_res = l2f.predict(
            test_x_across_task[9],
            representation=9,
            decider=9
        )
        uf[rep] = 1 - np.mean(
                test_y_across_task[9]==uf_res
            )

        posteriors_across_trees_hybrid_uf = estimate_posteriors(
            l2f_,
            test_x_across_task[9],
            representation=0,
            decider=0
            )

        hybrid_posterior_all = np.concatenate(
            (
                posteriors_across_trees[best_25_tree],
                posteriors_across_trees_hybrid_uf[0]
            ),
            axis=0
        )
        hybrid_posterior = np.mean(
            hybrid_posterior_all,
            axis=0
        )
        hybrid_res = np.argmax(hybrid_posterior,axis=1) + 90
        hybrid[rep] = 1 - np.mean(
                test_y_across_task[9]==hybrid_res
            )
    mean_accuracy_dict['hybrid'].extend(np.mean(hybrid))
    std_accuracy_dict['hybrid'].extend(hybrid,ddof=1)

    mean_accuracy_dict['building'].extend(np.mean(building))
    std_accuracy_dict['building'].extend(building,ddof=1)

    mean_accuracy_dict['recruiting'].extend(np.mean(recruiting))
    std_accuracy_dict['hybrid'].extend(recruiting,ddof=1)

    mean_accuracy_dict['UF'].extend(np.mean(uf))
    std_accuracy_dict['hybrid'].extend(uf,ddof=1)

summary = (mean_accuracy_dict,std_accuracy_dict)

with open('result/recruitment_exp.pickle','wb') as f:
    pickle.dump(summary,f)
# %%
