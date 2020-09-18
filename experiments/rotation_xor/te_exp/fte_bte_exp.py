#%%
import random
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd

import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil

from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import TreeClassificationTransformer, NeuralClassificationTransformer
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter

from joblib import Parallel, delayed
from multiprocessing import Pool

#%%
def LF_experiment(num_task_1_data, rep):

    default_transformer_class = TreeClassificationTransformer
    default_transformer_kwargs = {"kwargs" : {"max_depth" : 30}}

    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = {}

    default_decider_class = SimpleArgmaxAverage
    default_decider_kwargs = {}
    progressive_learner = ProgressiveLearner(default_transformer_class = default_transformer_class,
                                         default_transformer_kwargs = default_transformer_kwargs,
                                         default_voter_class = default_voter_class,
                                         default_voter_kwargs = default_voter_kwargs,
                                         default_decider_class = default_decider_class)

    X_train_task0, y_train_task0 = generate_gaussian_parity(n = num_task_1_data, angle_params = 0, acorn=1)
    X_train_task1, y_train_task1 = generate_gaussian_parity(n = 100, angle_params = 10, acorn=1)
    X_test_task0, y_test_task0 = generate_gaussian_parity(n = 10000, angle_params = 0, acorn=2)


    progressive_learner.add_task(
        X_train_task0,
        y_train_task0,
        num_transformers = 10,
        transformer_voter_decider_split = [0.67, 0.33, 0],
        decider_kwargs = {"classes" : np.unique(y_train_task0)}
        )
    llf_task=progressive_learner.predict(X_test_task0, task_id=0)
    single_task_accuracy = np.nanmean(llf_task == y_test_task0)
    single_task_error = 1 - single_task_accuracy

    progressive_learner.add_transformer(
            X = X_train_task1,
            y = y_train_task1,
            transformer_data_proportion = 1,
            num_transformers = 10,
            backward_task_ids = [0]
        )

    llf_task=progressive_learner.predict(X_test_task0, task_id=0)
    double_task_accuracy = np.nanmean(llf_task == y_test_task0)
    double_task_error = 1 - double_task_accuracy

    if double_task_error == 0 or single_task_error == 0:
        te = 1
    else:
        te = (single_task_error + 1e-6) / (double_task_error + 1e-6)

    df = pd.DataFrame()
    df['te'] = [te]

    print('n = {}, te = {}'.format(num_task_1_data, te))
    file_to_save = 'result/'+str(num_task_1_data)+'_'+str(rep)+'.pickle'
    with open(file_to_save, 'wb') as f:
        pickle.dump(df, f)

#%%
def generate_gaussian_parity(n, cov_scale=1, angle_params=None, k=1, acorn=None):
    means = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    blob = np.concatenate([np.random.multivariate_normal(mean, cov_scale * np.eye(len(mean)),
                                                 size=int(n / 4)) for mean in means])

    X = np.zeros_like(blob)
    Y = np.logical_xor(blob[:, 0] > 0, blob[:, 1] > 0)
    X[:, 0] = blob[:, 0] * np.cos(angle_params * np.pi / 180) + blob[:, 1] * np.sin(angle_params * np.pi / 180)
    X[:, 1] = -blob[:, 0] * np.sin(angle_params * np.pi / 180) + blob[:, 1] * np.cos(angle_params * np.pi / 180)
    return X, Y.astype(int)

#%%
num_task_1_data_ra=(2**np.arange(np.log2(60), np.log2(5010)+1, .25)).astype('int')
reps = range(1000)
iterable = product(num_task_1_data_ra, reps)
Parallel(n_jobs=-1,verbose=1)(delayed(LF_experiment)(num_task_1_data, rep) for num_task_1_data, rep in iterable)




# %%
