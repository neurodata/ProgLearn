import numpy as np
import random

from math import log2, ceil

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from proglearn.forest import LifelongClassificationForest, UncertaintyForest
from proglearn.sims import *
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import (
    SimpleArgmaxAverage,
    KNNClassificationDecider,
    KNNSimpleClassificationDecider,
)
from proglearn.transformers import (
    TreeClassificationTransformer,
    NeuralClassificationTransformer,
)
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter

from joblib import Parallel, delayed


def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c


def plot_xor_rxor(data, labels, title):
    colors = sns.color_palette("Dark2", n_colors=2)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(data[:, 0], data[:, 1], c=get_colors(colors, labels), s=50)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=30)
    # plt.tight_layout()
    ax.axis("off")
    plt.show()


def run(mc_rep, n_test, n_trees, n_xor, n_rxor, mean_error, std_error, mean_te, std_te):
    for i, n1 in enumerate(n_xor):
        # print('starting to compute %s xor\n'%n1)
        error = np.array(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(experiment)(
                    n1, 0, task2_angle=np.pi / 4, max_depth=ceil(log2(n1))
                )
                for _ in range(mc_rep)
            )
        )
        mean_error[:, i] = np.mean(error, axis=0)
        std_error[:, i] = np.std(error, ddof=1, axis=0)
        mean_te[0, i] = np.mean(error[:, 0]) / np.mean(error[:, 1])
        mean_te[1, i] = np.mean(error[:, 2]) / np.mean(error[:, 3])
        mean_te[2, i] = np.mean(error[:, 0]) / np.mean(error[:, 4])
        mean_te[3, i] = np.mean(error[:, 2]) / np.mean(error[:, 5])

        if n1 == n_xor[-1]:
            for j, n2 in enumerate(n_rxor):
                # print('starting to compute %s rxor\n'%n2)

                error = np.array(
                    Parallel(n_jobs=-1, verbose=0)(
                        delayed(experiment)(
                            n1, n2, task2_angle=np.pi / 4, max_depth=ceil(log2(750))
                        )
                        for _ in range(mc_rep)
                    )
                )
                mean_error[:, i + j + 1] = np.mean(error, axis=0)
                std_error[:, i + j + 1] = np.std(error, ddof=1, axis=0)
                mean_te[0, i + j + 1] = np.mean(error[:, 0]) / np.mean(error[:, 1])
                mean_te[1, i + j + 1] = np.mean(error[:, 2]) / np.mean(error[:, 3])
                mean_te[2, i + j + 1] = np.mean(error[:, 0]) / np.mean(error[:, 4])
                mean_te[3, i + j + 1] = np.mean(error[:, 2]) / np.mean(error[:, 5])

    return mean_error, std_error, mean_te, std_te


def experiment(
    n_task1,
    n_task2,
    n_test=1000,
    task1_angle=0,
    task2_angle=np.pi / 2,
    n_trees=10,
    max_depth=None,
    random_state=None,
    simple=True,
    k_neighbors=5,
):
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
        
    simple : boolean, default=True
        Determines whether to use KNNSimpleClassificationDecider or
        KNNClassificationDecider.
        
    k_neighbors : int, default=5
        Determines how many neighbors to use for KNN.
        
    Returns
    -------
    errors : array of shape [6]
        Elements of the array is organized as single task error task1,
        multitask error task1, single task error task2,
        multitask error task2, naive UF error task1,
        naive UF task2.
    """

    if n_task1 == 0 and n_task2 == 0:
        raise ValueError("Wake up and provide samples to train!!!")

    if random_state != None:
        np.random.seed(random_state)

    errors = np.zeros(6, dtype=float)

    default_transformer_class = TreeClassificationTransformer
    default_transformer_kwargs = {"kwargs": {"max_depth": max_depth}}

    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = {}

    default_decider_class = SimpleArgmaxAverage
    default_decider_kwargs = {"classes": np.arange(2)}

    if simple:
        KNN_class = KNNSimpleClassificationDecider
        KNN_decider_kwargs = {"classes": np.arange(2), "k": k_neighbors}
    else:
        KNN_class = KNNClassificationDecider
        KNN_decider_kwargs = {"classes": np.arange(2), "k": k_neighbors}

    progressive_learner = ProgressiveLearner(
        default_transformer_class=default_transformer_class,
        default_transformer_kwargs=default_transformer_kwargs,
        default_voter_class=default_voter_class,
        default_voter_kwargs=default_voter_kwargs,
        default_decider_class=KNN_class,
        default_decider_kwargs=KNN_decider_kwargs,
    )
    uf = ProgressiveLearner(
        default_transformer_class=default_transformer_class,
        default_transformer_kwargs=default_transformer_kwargs,
        default_voter_class=default_voter_class,
        default_voter_kwargs=default_voter_kwargs,
        default_decider_class=default_decider_class,
        default_decider_kwargs=default_decider_kwargs,
    )
    naive_uf = ProgressiveLearner(
        default_transformer_class=default_transformer_class,
        default_transformer_kwargs=default_transformer_kwargs,
        default_voter_class=default_voter_class,
        default_voter_kwargs=default_voter_kwargs,
        default_decider_class=default_decider_class,
        default_decider_kwargs=default_decider_kwargs,
    )

    # source data
    X_task1, y_task1 = generate_gaussian_parity(n_task1, angle_params=task1_angle)
    test_task1, test_label_task1 = generate_gaussian_parity(
        n_test, angle_params=task1_angle
    )

    # target data
    X_task2, y_task2 = generate_gaussian_parity(n_task2, angle_params=task2_angle)
    test_task2, test_label_task2 = generate_gaussian_parity(
        n_test, angle_params=task2_angle
    )

    if n_task1 == 0:
        progressive_learner.add_task(
            X_task2,
            y_task2,
            num_transformers=n_trees,
            transformer_voter_decider_split=[0.33, 0.33, 0.33],
        )

        errors[0] = 0.5
        errors[1] = 0.5

        uf_task2 = progressive_learner.predict(
            test_task2, transformer_ids=[0], task_id=0
        )
        l2f_task2 = progressive_learner.predict(test_task2, task_id=0)

        errors[2] = 1 - np.mean(uf_task2 == test_label_task2)
        errors[3] = 1 - np.mean(l2f_task2 == test_label_task2)

        errors[4] = 0.5
        errors[5] = 1 - np.mean(uf_task2 == test_label_task2)
    elif n_task2 == 0:
        progressive_learner.add_task(
            X_task1,
            y_task1,
            num_transformers=n_trees,
            transformer_voter_decider_split=[0.33, 0.33, 0.33],
        )

        uf_task1 = progressive_learner.predict(
            test_task1, transformer_ids=[0], task_id=0
        )
        l2f_task1 = progressive_learner.predict(test_task1, task_id=0)

        errors[0] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[1] = 1 - np.mean(l2f_task1 == test_label_task1)

        errors[2] = 0.5
        errors[3] = 0.5

        errors[4] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[5] = 0.5
    else:
        progressive_learner.add_task(
            X_task1,
            y_task1,
            num_transformers=n_trees,
            transformer_voter_decider_split=[0.33, 0.33, 0.33],
        )
        progressive_learner.add_task(
            X_task2,
            y_task2,
            num_transformers=n_trees,
            transformer_voter_decider_split=[0.33, 0.33, 0.33],
        )

        uf.add_task(X_task1, y_task1, num_transformers=2 * n_trees)
        uf.add_task(X_task2, y_task2, num_transformers=2 * n_trees)

        naive_uf_train_x = np.concatenate((X_task1, X_task2), axis=0)
        naive_uf_train_y = np.concatenate((y_task1, y_task2), axis=0)
        naive_uf.add_task(naive_uf_train_x, naive_uf_train_y, num_transformers=n_trees)

        uf_task1 = uf.predict(test_task1, transformer_ids=[0], task_id=0)
        l2f_task1 = progressive_learner.predict(test_task1, task_id=0)
        uf_task2 = uf.predict(test_task2, transformer_ids=[1], task_id=1)
        l2f_task2 = progressive_learner.predict(test_task2, task_id=1)
        naive_uf_task1 = naive_uf.predict(test_task1, transformer_ids=[0], task_id=0)
        naive_uf_task2 = naive_uf.predict(test_task2, transformer_ids=[0], task_id=0)

        errors[0] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[1] = 1 - np.mean(l2f_task1 == test_label_task1)
        errors[2] = 1 - np.mean(uf_task2 == test_label_task2)
        errors[3] = 1 - np.mean(l2f_task2 == test_label_task2)
        errors[4] = 1 - np.mean(naive_uf_task1 == test_label_task1)
        errors[5] = 1 - np.mean(naive_uf_task2 == test_label_task2)

    return errors


def experiment_KNN_BTE(
    n_task1,
    n_task2,
    n_test=1000,
    angle_sweep=range(0, 45, 5),
    n_trees=10,
    max_depth=None,
    k_neighbors=5,
    random_state=None,
    simple=0,
):
    """
    A function to do progressive experiment between two tasks
    where the task data is generated using Gaussian parity.
    Tests specifically the BTE using the first index in the
    angle_sweep as the first task and the other indices as
    other tasks.
    
    Parameters
    ----------
    n_task1 : int
        Total number of train sample for task 1.
    
    n_task2 : int
        Total number of train dsample for task 2

    n_test : int, optional (default=1000)
        Number of test sample for each task.
        
    angle_sweep : range (default=range(0,45,5))
        Angles to sweep through. Task 1 is the first in the list,
        other tasks are the following angles in the list.
            
    n_trees : int, optional (default=10)
        Number of total trees to train for each task.

    k_neighbors : int, (default=5)
        Determines how many neighbors to use for KNN.

    max_depth : int, optional (default=None)
        Maximum allowable depth for each tree.
        
    random_state : int, optional (default=None)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        
    simple : int, default=0
        Determines which decider to use.
            0 - SimpleArgmaxAverage
            1 - KNNSimpleClassificationDecider
            2 - KNNClassificationDecider

    transformer_voter_decider_split : list, default=[0.33, 0.33, 0.33]
        Determines how data is split for training.
        
    Returns
    -------
    errors : array of shape [1, len(range(0,45,5))]
        Elements of the array is organized as the errors from adding
        successive tasks.
    """

    # See if the sample sizes for both training sets are given.
    if n_task1 == 0 and n_task2 == 0:
        raise ValueError("Wake up and provide samples to train!!!")

    # If acorn is specified, set random seed to it.
    if random_state != None:
        np.random.seed(random_state)

    # Initialize array for storing errors, task 1, and task 2.
    errors = np.zeros(len(angle_sweep), dtype=float)

    # Initialize the transformer type and arguments.
    default_transformer_class = TreeClassificationTransformer
    default_transformer_kwargs = {"kwargs": {"max_depth": max_depth}}

    # Initialize the voter type and arguments.
    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = {"classes": np.arange(2)}

    # Initialize the decider type and arguments.
    if simple == 0:
        default_decider_class = SimpleArgmaxAverage
        default_decider_kwargs = {"classes": np.arange(2)}
    elif simple == 1:
        default_decider_class = KNNClassificationDecider
        default_decider_kwargs = {"classes": np.arange(2), "k": k_neighbors}
    elif simple == 2:
        default_decider_class = KNNSimpleClassificationDecider
        default_decider_kwargs = {"classes": np.arange(2), "k": k_neighbors}

    # Initialize the progressive learner using the transformer, voter and decider classes.
    progressive_learner = ProgressiveLearner(
        default_transformer_class=default_transformer_class,
        default_transformer_kwargs=default_transformer_kwargs,
        default_voter_class=default_voter_class,
        default_voter_kwargs=default_voter_kwargs,
        default_decider_class=default_decider_class,
        default_decider_kwargs=default_decider_kwargs,
    )

    # Create the datasets with the Gaussian mean for task 1.
    X_task1, y_task1 = generate_gaussian_parity(n_task1, angle_params=angle_sweep[0])
    test_task1, test_label_task1 = generate_gaussian_parity(
        n_task1, angle_params=angle_sweep[0]
    )

    # Add a task for the task 1, predict the probabilities and add the MSE to the error array.
    progressive_learner.add_task(X_task1, y_task1, transformer_voter_decider_split=[0.33, 0.33, 0.33])

    l2f_task1 = progressive_learner.predict(test_task1, task_id=0)
    errors[0] = np.mean(l2f_task1 == test_label_task1)

    # Then, add the transformer trained on task 2, predict and add the MSE to the error array.
    for i in range(len(angle_sweep) - 1):
        X2, Z2 = generate_gaussian_parity(n_task2, angle_params=angle_sweep[i + 1])
        progressive_learner.add_task(X_task1, y_task1, transformer_voter_decider_split=[0.33, 0.33, 0.33])

        predicted_transformer_Z1 = progressive_learner.predict(test_task1, task_id=0)
        errors[i + 1] = np.mean(predicted_transformer_Z1 == test_label_task1)

    return errors


def bte_v_angle(
    angle_sweep, task1_sample, task2_sample, mc_rep, acorn=None, simple=True
):
    mean_te = np.zeros(len(angle_sweep), dtype=float)
    mean_te_uf = np.zeros(len(angle_sweep), dtype=float)

    for ii, angle in enumerate(angle_sweep):
        error = np.array(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(experiment)(
                    task1_sample,
                    task2_sample,
                    task2_angle=angle * np.pi / 180,
                    max_depth=ceil(log2(task1_sample)),
                    random_state=acorn,
                    simple=simple,
                )
                for _ in range(mc_rep)
            )
        )

        mean_te[ii] = np.mean(error[0, :]) / np.mean(error[1, :])
        mean_te_uf[ii] = np.mean(error[0, :]) / np.mean(error[4, :])
    return mean_te, mean_te_uf
