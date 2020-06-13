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

# %%
def predict_proba(voter, nodes_across_trees):
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
                