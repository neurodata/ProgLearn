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

from joblib import Parallel, delayed
from multiprocessing import Pool

import sys
sys.path.append("../../../proglearn/")
from progressive_learner import ProgressiveLearner
from deciders import SimpleAverage
from transformers import TreeClassificationTransformer, NeuralClassificationTransformer 
from voters import TreeClassificationVoter, KNNClassificationVoter

def generate_gaussian_parity(n, cov_scale=1, angle_params=None, k=1, acorn=None):
    means = [[-1, -1], [-1, 1], [1, -1], [1, 1]]    
    blob = np.concatenate([np.random.multivariate_normal(mean, cov_scale * np.eye(len(mean)), 
                                                 size=int(n / 4)) for mean in means])

    X = np.zeros_like(blob)
    Y = np.logical_xor(blob[:, 0] > 0, blob[:, 1] > 0)
    X[:, 0] = blob[:, 0] * np.cos(angle_params * np.pi / 180) + blob[:, 1] * np.sin(angle_params * np.pi / 180)
    X[:, 1] = -blob[:, 0] * np.sin(angle_params * np.pi / 180) + blob[:, 1] * np.cos(angle_params * np.pi / 180)
    return X, Y.astype(int)


def LF_experiment(angle, reps=1, ntrees=10, acorn=None):
    
    errors = np.zeros(2)
    
    for rep in range(reps):
        print("Starting Rep {} of Angle {}".format(rep, angle))
        X_base_train, y_base_train = generate_gaussian_parity(n = 100, angle_params = 0, acorn=rep)
        X_base_test, y_base_test = generate_gaussian_parity(n = 10000, angle_params = 0, acorn=rep)
        X_rotated_train, y_rotated_train = generate_gaussian_parity(n = 100, angle_params = angle, acorn=rep)
        
    
        default_transformer_class = TreeClassificationTransformer
        default_transformer_kwargs = {"kwargs" : {"max_depth" : 10}}

        default_voter_class = TreeClassificationVoter
        default_voter_kwargs = {}

        default_decider_class = SimpleAverage
        default_decider_kwargs = {}
        progressive_learner = ProgressiveLearner(default_transformer_class = default_transformer_class, 
                                             default_transformer_kwargs = default_transformer_kwargs,
                                             default_voter_class = default_voter_class,
                                             default_voter_kwargs = default_voter_kwargs,
                                             default_decider_class = default_decider_class)
        progressive_learner.add_task(
            X_base_train, 
            y_base_train,
            num_transformers = ntrees,
            transformer_voter_decider_split = [0.67, 0.33, 0],
            decider_kwargs = {"classes" : np.unique(y_base_train)}
        )
        base_predictions_test = progressive_learner.predict(X_base_test, task_id=0)
        progressive_learner.add_transformer(
            X = X_rotated_train, 
            y = y_rotated_train,
            transformer_data_proportion = 1,
            num_transformers = 10,
            backward_task_ids = [0]
        )

        all_predictions_test=progressive_learner.predict(X_base_test, task_id=0)

        errors[1] = errors[1]+(1 - np.mean(all_predictions_test == y_base_test))
        errors[0] = errors[0]+(1 - np.mean(base_predictions_test == y_base_test))

    
    errors = errors/reps
    print("Errors For Angle {}: {}".format(angle, errors))
    with open('results/angle_'+str(angle)+'.pickle', 'wb') as f:
        pickle.dump(errors, f, protocol = 2)
        

### MAIN HYPERPARAMS ###
granularity = 1
reps = 10
########################


def perform_angle(angle):
    LF_experiment(angle, reps=reps, ntrees=10)

angles = np.arange(0,90 + granularity,granularity)
Parallel(n_jobs=-1, verbose = 1)(delayed(LF_experiment)(angle, reps=reps, ntrees=10) for angle in angles)