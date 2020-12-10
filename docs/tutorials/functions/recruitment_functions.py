import numpy as np
from math import log2, ceil
import random

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

from proglearn.progressive_learner import ClassificationProgressiveLearner
from proglearn.transformers import TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter
from proglearn.deciders import SimpleArgmaxAverage


class PosteriorsByTreeLearner(ClassificationProgressiveLearner):
    """
    Variation on the progressive learner class ClassificationProgressiveLearner
    to allow for return of posterior probabilities by tree.
    """

    def predict_proba(self, X, task_id, transformer_ids=None):
        """
        Calls predict_proba_tree for the decider class. Works in conjunction with
        class PosteriorsByTree.
        """
        decider = self.task_id_to_decider[task_id]
        return self.task_id_to_decider[task_id].predict_proba_tree(
            X, transformer_ids=transformer_ids
        )


class PosteriorsByTree(SimpleArgmaxAverage):
    """Variation on the decider class SimpleArgmaxAverage to allow for return of
    posterior probabilities by tree.
    """

    def predict_proba_tree(self, X, transformer_ids=None):
        """
        Predicts posterior probabilities by tree.

        Returns array of dimension (num_transformers*len(transformer_ids))
        containing posterior probabilities.
        """
        vote_per_alltrees = []
        for transformer_id in (
            transformer_ids
            if transformer_ids is not None
            else self.transformer_id_to_voters_.keys()
        ):
            for bag_id in range(
                len(self.transformer_id_to_transformers_[transformer_id])
            ):
                transformer = self.transformer_id_to_transformers_[transformer_id][
                    bag_id
                ]
                X_transformed = transformer.transform(X)
                voter = self.transformer_id_to_voters_[transformer_id][bag_id]
                vote = voter.predict_proba(X_transformed)
                vote_per_alltrees.append(vote)
        return vote_per_alltrees


def sort_data(data_x, data_y, num_points_per_task, total_task=10, shift=1):
    """
    Sorts data into training and testing data sets for each task.
    """

    # get data and indices
    x = data_x.copy()
    y = data_y.copy()
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]

    # initialize lists
    train_x_across_task = []
    train_y_across_task = []
    test_x_across_task = []
    test_y_across_task = []

    # calculate amount of samples per task
    batch_per_task = 5000 // num_points_per_task
    sample_per_class = num_points_per_task // total_task
    test_data_slot = 100 // batch_per_task

    # sort data into batches per class
    for task in range(total_task):
        for batch in range(batch_per_task):
            for class_no in range(task * 10, (task + 1) * 10, 1):
                indx = np.roll(idx[class_no], (shift - 1) * 100)
                # if first batch, reset arrays with new data ; otherwise, concatenate arrays
                if batch == 0 and class_no == task * 10:
                    train_x = x[
                        indx[batch * sample_per_class : (batch + 1) * sample_per_class],
                        :,
                    ]
                    train_y = y[
                        indx[batch * sample_per_class : (batch + 1) * sample_per_class]
                    ]
                    test_x = x[
                        indx[
                            batch * test_data_slot
                            + 500 : (batch + 1) * test_data_slot
                            + 500
                        ],
                        :,
                    ]
                    test_y = y[
                        indx[
                            batch * test_data_slot
                            + 500 : (batch + 1) * test_data_slot
                            + 500
                        ]
                    ]
                else:
                    train_x = np.concatenate(
                        (
                            train_x,
                            x[
                                indx[
                                    batch
                                    * sample_per_class : (batch + 1)
                                    * sample_per_class
                                ],
                                :,
                            ],
                        ),
                        axis=0,
                    )
                    train_y = np.concatenate(
                        (
                            train_y,
                            y[
                                indx[
                                    batch
                                    * sample_per_class : (batch + 1)
                                    * sample_per_class
                                ]
                            ],
                        ),
                        axis=0,
                    )
                    test_x = np.concatenate(
                        (
                            test_x,
                            x[
                                indx[
                                    batch * test_data_slot
                                    + 500 : (batch + 1) * test_data_slot
                                    + 500
                                ],
                                :,
                            ],
                        ),
                        axis=0,
                    )
                    test_y = np.concatenate(
                        (
                            test_y,
                            y[
                                indx[
                                    batch * test_data_slot
                                    + 500 : (batch + 1) * test_data_slot
                                    + 500
                                ]
                            ],
                        ),
                        axis=0,
                    )

        # append data to lists
        train_x_across_task.append(train_x)
        train_y_across_task.append(train_y)
        test_x_across_task.append(test_x)
        test_y_across_task.append(test_y)

    return (
        train_x_across_task,
        train_y_across_task,
        test_x_across_task,
        test_y_across_task,
    )


def experiment(
    train_x_across_task,
    train_y_across_task,
    test_x_across_task,
    test_y_across_task,
    ntrees,
    reps,
    estimation_set,
    num_points_per_task,
    num_points_per_forest,
    task_10_sample,
):
    """
    Run the recruitment experiment.
    """

    # create matrices for storing values
    hybrid = np.zeros(reps, dtype=float)
    building = np.zeros(reps, dtype=float)
    recruiting = np.zeros(reps, dtype=float)
    uf = np.zeros(reps, dtype=float)
    mean_accuracy_dict = {"building": [], "UF": [], "recruiting": [], "hybrid": []}
    std_accuracy_dict = {"building": [], "UF": [], "recruiting": [], "hybrid": []}

    # iterate over all sample sizes ns
    for ns in task_10_sample:

        # size of estimation and validation sample sets
        estimation_sample_no = ceil(estimation_set * ns)
        validation_sample_no = ns - estimation_sample_no

        # repeat `rep` times
        for rep in range(reps):
            print("doing {} samples for {} th rep".format(ns, rep))

            # initiate lifelong learner
            l2f = PosteriorsByTreeLearner(
                default_transformer_class=TreeClassificationTransformer,
                default_transformer_kwargs={},
                default_voter_class=TreeClassificationVoter,
                default_voter_kwargs={"finite_sample_correction": False},
                default_decider_class=PosteriorsByTree,
                default_decider_kwargs={},
            )

            # train l2f on first 9 tasks
            for task in range(9):
                indx = np.random.choice(
                    num_points_per_task, num_points_per_forest, replace=False
                )
                cur_X = train_x_across_task[task][indx]
                cur_y = train_y_across_task[task][indx]

                l2f.add_task(
                    cur_X,
                    cur_y,
                    num_transformers=ntrees,
                    # transformer_kwargs={"kwargs":{"max_depth": ceil(log2(num_points_per_forest))}},
                    voter_kwargs={
                        "classes": np.unique(cur_y),
                        "finite_sample_correction": False,
                    },
                    decider_kwargs={"classes": np.unique(cur_y)},
                )

            # train l2f on 10th task
            task_10_train_indx = np.random.choice(
                num_points_per_task, ns, replace=False
            )
            cur_X = train_x_across_task[9][task_10_train_indx[:estimation_sample_no]]
            cur_y = train_y_across_task[9][task_10_train_indx[:estimation_sample_no]]

            l2f.add_task(
                cur_X,
                cur_y,
                num_transformers=ntrees,
                # transformer_kwargs={"kwargs":{"max_depth": ceil(log2(estimation_sample_no))}},
                voter_kwargs={
                    "classes": np.unique(cur_y),
                    "finite_sample_correction": False,
                },
                decider_kwargs={"classes": np.unique(cur_y)},
            )

            ## L2F validation ####################################
            # get posteriors for l2f on first 9 tasks
            # want posteriors_across_trees to have shape (9*ntrees, validation_sample_no, 10)
            posteriors_across_trees = l2f.predict_proba(
                train_x_across_task[9][task_10_train_indx[estimation_sample_no:]],
                task_id=9,
                transformer_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            )
            # compare error in each tree and choose best 25/50 trees
            error_across_trees = np.zeros(9 * ntrees)
            validation_target = train_y_across_task[9][
                task_10_train_indx[estimation_sample_no:]
            ]
            for tree in range(len(posteriors_across_trees)):
                res = np.argmax(posteriors_across_trees[tree], axis=1) + 90
                error_across_trees[tree] = 1 - np.mean(validation_target == res)
            best_50_tree = np.argsort(error_across_trees)[:50]
            best_25_tree = best_50_tree[:25]

            ## uf trees validation ###############################
            # get posteriors for l2f on only the 10th task
            posteriors_across_trees = l2f.predict_proba(
                train_x_across_task[9][task_10_train_indx[estimation_sample_no:]],
                task_id=9,
                transformer_ids=[9],
            )
            # compare error in each tree and choose best 25 trees
            error_across_trees = np.zeros(ntrees)
            validation_target = train_y_across_task[9][
                task_10_train_indx[estimation_sample_no:]
            ]
            for tree in range(ntrees):
                res = np.argmax(posteriors_across_trees[tree], axis=1) + 90
                error_across_trees[tree] = 1 - np.mean(validation_target == res)
            best_25_uf_tree = np.argsort(error_across_trees)[:25]

            ## evaluation ########################################
            # train 10th tree under each scenario: building, recruiting, hybrid, UF
            # BUILDING
            building_res = l2f.predict(test_x_across_task[9], task_id=9)
            building[rep] = 1 - np.mean(test_y_across_task[9] == building_res)
            # UF
            uf_res = l2f.predict(test_x_across_task[9], task_id=9, transformer_ids=[9])
            uf[rep] = 1 - np.mean(test_y_across_task[9] == uf_res)
            # RECRUITING
            posteriors_across_trees = l2f.predict_proba(
                test_x_across_task[9],
                task_id=9,
                transformer_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            )
            recruiting_posterior = np.mean(
                np.array(posteriors_across_trees)[best_50_tree], axis=0
            )
            res = np.argmax(recruiting_posterior, axis=1) + 90
            recruiting[rep] = 1 - np.mean(test_y_across_task[9] == res)
            # HYBRID
            posteriors_across_trees_hybrid_uf = l2f.predict_proba(
                test_x_across_task[9], task_id=9, transformer_ids=[9]
            )
            hybrid_posterior_all = np.concatenate(
                (
                    np.array(posteriors_across_trees)[best_25_tree],
                    np.array(posteriors_across_trees_hybrid_uf)[best_25_uf_tree],
                ),
                axis=0,
            )
            hybrid_posterior = np.mean(hybrid_posterior_all, axis=0)
            hybrid_res = np.argmax(hybrid_posterior, axis=1) + 90
            hybrid[rep] = 1 - np.mean(test_y_across_task[9] == hybrid_res)

        # print(np.mean(building))
        # print(np.mean(uf))
        # print(np.mean(recruiting))
        # print(np.mean(hybrid))

        # calculate mean and stdev for each
        mean_accuracy_dict["building"].append(np.mean(building))
        std_accuracy_dict["building"].append(np.std(building, ddof=1))
        mean_accuracy_dict["UF"].append(np.mean(uf))
        std_accuracy_dict["UF"].append(np.std(uf, ddof=1))
        mean_accuracy_dict["recruiting"].append(np.mean(recruiting))
        std_accuracy_dict["recruiting"].append(np.std(recruiting, ddof=1))
        mean_accuracy_dict["hybrid"].append(np.mean(hybrid))
        std_accuracy_dict["hybrid"].append(np.std(hybrid, ddof=1))

    return mean_accuracy_dict, std_accuracy_dict


def recruitment_plot(mean_acc_dict, std_acc_dict, ns):
    """
    Plot the results from the recruitment experiment.
    """

    # determine colors and labels for figure
    colors = sns.color_palette("Set1", n_colors=len(mean_acc_dict))
    labels = ["L2F (building)", "UF (new)", "recruiting", "hybrid"]

    # plot and format figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for i, key in enumerate(mean_acc_dict):
        ax.plot(ns, mean_acc_dict[key], c=colors[i], label=labels[i])
        upper_bound = np.array(mean_acc_dict[key]) + 1.96 * np.array(std_acc_dict[key])
        lower_bound = np.array(mean_acc_dict[key]) - 1.96 * np.array(std_acc_dict[key])
        ax.fill_between(
            ns,
            upper_bound,
            lower_bound,
            where=upper_bound >= lower_bound,
            facecolor=colors[i],
            alpha=0.15,
            interpolate=False,
        )
    # ax.set_title("CIFAR Recruitment Experiment",fontsize=30)
    ax.set_ylabel("Generalization Error (Task 10)", fontsize=28)
    ax.set_xlabel("Number of Task 10 Samples", fontsize=30)
    ax.tick_params(labelsize=28)
    ax.set_xscale("log")
    ax.set_xticks([100, 500, 5000])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_ylim(0.450, 0.825)
    ax.set_yticks([0.45, 0.55, 0.65, 0.75])
    ax.legend(fontsize=12)
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    plt.tight_layout()


def sort_data_mnist(
    x_train,
    y_train,
    x_test,
    y_test,
    num_points_per_task,
    test_points_per_task,
    n_tasks=10,
    shift=1,
):
    """
    Sorts data into training and testing data sets for each task.
    Generalized for MNIST datasets and different task numbers.
    """

    # reformat data
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    # get data and indices
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    idx_train = [np.where(y_train == u)[0] for u in unique_labels]
    idx_test = [np.where(y_test == u)[0] for u in unique_labels]
    idx_len = [len(i) for i in idx_train]

    # initialize lists
    train_x_across_task = []
    train_y_across_task = []
    test_x_across_task = []
    test_y_across_task = []

    # calculate amount of samples per task
    labels_per_task = len(unique_labels) // n_tasks
    # print(labels_per_task)
    train_per_class = num_points_per_task // labels_per_task
    # print(train_per_class)
    test_per_class = test_points_per_task
    # print(test_per_class)

    # sort data into batches per class
    for task in range(n_tasks):
        for class_no in range(task * labels_per_task, (task + 1) * labels_per_task, 1):
            indx_train = np.roll(
                idx_train[class_no], (shift - 1) * n_tasks * labels_per_task
            )
            indx_test = np.roll(
                idx_test[class_no], (shift - 1) * n_tasks * labels_per_task
            )
            # if first batch, reset arrays with new data ; otherwise, concatenate arrays
            if class_no == (task * labels_per_task):
                xtr = x_train[indx_train[0:train_per_class], :]
                ytr = y_train[indx_train[0:train_per_class]].flatten()
                xte = x_test[indx_test[0:test_per_class], :]
                yte = y_test[indx_test[0:test_per_class]].flatten()
            else:
                xtr = np.concatenate(
                    (xtr, x_train[indx_train[0:train_per_class], :]), axis=0
                )
                ytr = np.concatenate(
                    (ytr, y_train[indx_train[0:train_per_class]].flatten()), axis=0
                )
                xte = np.concatenate(
                    (xte, x_test[indx_test[0:test_per_class], :]), axis=0
                )
                yte = np.concatenate(
                    (yte, y_test[indx_test[0:test_per_class]].flatten()), axis=0
                )

        # append data to lists
        train_x_across_task.append(xtr)
        train_y_across_task.append(ytr)
        test_x_across_task.append(xte)
        test_y_across_task.append(yte)

    return (
        train_x_across_task,
        train_y_across_task,
        test_x_across_task,
        test_y_across_task,
    )


def experiment_mnist(
    train_x_across_task,
    train_y_across_task,
    test_x_across_task,
    test_y_across_task,
    ntrees,
    n_tasks,
    reps,
    estimation_set,
    num_points_per_task,
    num_points_per_forest,
    task_10_sample,
):
    """
    Run the recruitment experiment.
    Generalized for MNIST datasets and different task numbers.
    """

    # create matrices for storing values
    hybrid = np.zeros(reps, dtype=float)
    building = np.zeros(reps, dtype=float)
    recruiting = np.zeros(reps, dtype=float)
    uf = np.zeros(reps, dtype=float)
    mean_accuracy_dict = {"building": [], "UF": [], "recruiting": [], "hybrid": []}
    std_accuracy_dict = {"building": [], "UF": [], "recruiting": [], "hybrid": []}

    # calculate other params
    unique_labels = np.unique(train_y_across_task)
    labels_per_task = len(unique_labels) // n_tasks
    # print(labels_per_task)

    # iterate over all sample sizes ns
    for ns in task_10_sample:

        # size of estimation and validation sample sets
        estimation_sample_no = ceil(estimation_set * ns)
        validation_sample_no = ns - estimation_sample_no

        # repeat `rep` times
        for rep in range(reps):
            # print("doing {} samples for {} th rep".format(ns,rep))

            # initiate lifelong learner
            l2f = PosteriorsByTreeLearner(
                default_transformer_class=TreeClassificationTransformer,
                default_transformer_kwargs={},
                default_voter_class=TreeClassificationVoter,
                default_voter_kwargs={},
                default_decider_class=PosteriorsByTree,
                default_decider_kwargs={},
            )

            # train l2f on first 9 tasks
            for task in range(n_tasks - 1):
                indx = np.random.choice(
                    num_points_per_task, num_points_per_forest, replace=False
                )
                cur_X = train_x_across_task[task][indx]
                cur_y = train_y_across_task[task][indx]

                l2f.add_task(
                    cur_X,
                    cur_y,
                    num_transformers=ntrees,
                    # transformer_kwargs={"kwargs":{"max_depth": ceil(log2(num_points_per_forest))}},
                    voter_kwargs={"classes": np.unique(cur_y)},
                    decider_kwargs={"classes": np.unique(cur_y)},
                )

            # train l2f on 10th task
            task_10_train_indx = np.random.choice(
                num_points_per_task, ns, replace=False
            )
            cur_X = train_x_across_task[n_tasks - 1][
                task_10_train_indx[:estimation_sample_no]
            ]
            cur_y = train_y_across_task[n_tasks - 1][
                task_10_train_indx[:estimation_sample_no]
            ]

            l2f.add_task(
                cur_X,
                cur_y,
                num_transformers=ntrees,
                # transformer_kwargs={"kwargs":{"max_depth": ceil(log2(estimation_sample_no))}},
                voter_kwargs={"classes": np.unique(cur_y)},
                decider_kwargs={"classes": np.unique(cur_y)},
            )

            ## L2F validation ####################################
            # get posteriors for l2f on first 9 tasks
            # want posteriors_across_trees to have shape ((n_tasks-1)*ntrees, validation_sample_no, 10)
            posteriors_across_trees = l2f.predict_proba(
                train_x_across_task[n_tasks - 1][
                    task_10_train_indx[estimation_sample_no:]
                ],
                task_id=n_tasks - 1,
                transformer_ids=list(range(n_tasks - 1)),
            )
            # compare error in each tree and choose best 25/50 trees
            error_across_trees = np.zeros((n_tasks - 1) * ntrees)
            validation_target = train_y_across_task[n_tasks - 1][
                task_10_train_indx[estimation_sample_no:]
            ]
            for tree in range(len(posteriors_across_trees)):
                res = np.argmax(
                    posteriors_across_trees[tree], axis=1
                ) + labels_per_task * (n_tasks - 1)
                error_across_trees[tree] = 1 - np.mean(validation_target == res)
            best_50_tree = np.argsort(error_across_trees)[:50]
            best_25_tree = best_50_tree[:25]

            ## uf trees validation ###############################
            # get posteriors for l2f on only the 10th task
            posteriors_across_trees = l2f.predict_proba(
                train_x_across_task[n_tasks - 1][
                    task_10_train_indx[estimation_sample_no:]
                ],
                task_id=n_tasks - 1,
                transformer_ids=[n_tasks - 1],
            )
            # compare error in each tree and choose best 25 trees
            error_across_trees = np.zeros(ntrees)
            validation_target = train_y_across_task[n_tasks - 1][
                task_10_train_indx[estimation_sample_no:]
            ]
            for tree in range(ntrees):
                res = np.argmax(
                    posteriors_across_trees[tree], axis=1
                ) + labels_per_task * (n_tasks - 1)
                error_across_trees[tree] = 1 - np.mean(validation_target == res)
            best_25_uf_tree = np.argsort(error_across_trees)[:25]

            ## evaluation ########################################
            # train 10th tree under each scenario: building, recruiting, hybrid, UF
            # BUILDING
            building_res = l2f.predict(
                test_x_across_task[n_tasks - 1], task_id=n_tasks - 1
            )
            building[rep] = 1 - np.mean(test_y_across_task[n_tasks - 1] == building_res)
            # UF
            uf_res = l2f.predict(
                test_x_across_task[n_tasks - 1],
                task_id=n_tasks - 1,
                transformer_ids=[n_tasks - 1],
            )
            uf[rep] = 1 - np.mean(test_y_across_task[n_tasks - 1] == uf_res)
            # RECRUITING
            posteriors_across_trees = l2f.predict_proba(
                test_x_across_task[n_tasks - 1],
                task_id=n_tasks - 1,
                transformer_ids=list(range(n_tasks - 1)),
            )
            recruiting_posterior = np.mean(
                np.array(posteriors_across_trees)[best_50_tree], axis=0
            )
            res = np.argmax(recruiting_posterior, axis=1) + labels_per_task * (
                n_tasks - 1
            )
            recruiting[rep] = 1 - np.mean(test_y_across_task[n_tasks - 1] == res)
            # HYBRID
            posteriors_across_trees_hybrid_uf = l2f.predict_proba(
                test_x_across_task[n_tasks - 1],
                task_id=n_tasks - 1,
                transformer_ids=[n_tasks - 1],
            )
            hybrid_posterior_all = np.concatenate(
                (
                    np.array(posteriors_across_trees)[best_25_tree],
                    np.array(posteriors_across_trees_hybrid_uf)[best_25_uf_tree],
                ),
                axis=0,
            )
            hybrid_posterior = np.mean(hybrid_posterior_all, axis=0)
            hybrid_res = np.argmax(hybrid_posterior, axis=1) + labels_per_task * (
                n_tasks - 1
            )
            hybrid[rep] = 1 - np.mean(test_y_across_task[n_tasks - 1] == hybrid_res)

        # print(np.mean(building))
        # print(np.mean(uf))
        # print(np.mean(recruiting))
        # print(np.mean(hybrid))

        # calculate mean and stdev for each
        mean_accuracy_dict["building"].append(np.mean(building))
        std_accuracy_dict["building"].append(np.std(building, ddof=1))
        mean_accuracy_dict["UF"].append(np.mean(uf))
        std_accuracy_dict["UF"].append(np.std(uf, ddof=1))
        mean_accuracy_dict["recruiting"].append(np.mean(recruiting))
        std_accuracy_dict["recruiting"].append(np.std(recruiting, ddof=1))
        mean_accuracy_dict["hybrid"].append(np.mean(hybrid))
        std_accuracy_dict["hybrid"].append(np.std(hybrid, ddof=1))

    return mean_accuracy_dict, std_accuracy_dict


def recruitment_plot_mnist(mean_acc_dict, std_acc_dict, ns):
    """
    Plot the results from the recruitment experiment for individual MNIST datasets.
    """
    # determine colors and labels for figure
    colors = sns.color_palette("Set1", n_colors=len(mean_acc_dict))
    labels = ["L2F (building)", "UF (new)", "recruiting", "hybrid"]

    # plot and format figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for i, key in enumerate(mean_acc_dict):
        ax.plot(ns, mean_acc_dict[key], c=colors[i], label=labels[i])
        upper_bound = np.array(mean_acc_dict[key]) + 1.96 * np.array(std_acc_dict[key])
        lower_bound = np.array(mean_acc_dict[key]) - 1.96 * np.array(std_acc_dict[key])
        ax.fill_between(
            ns,
            upper_bound,
            lower_bound,
            where=upper_bound >= lower_bound,
            facecolor=colors[i],
            alpha=0.15,
            interpolate=False,
        )
    ax.set_title("MNIST Recruitment", fontsize=30)
    ax.set_ylabel("Generalization Error (Task 5)", fontsize=28)
    ax.set_xlabel("Number of Task 5 Samples", fontsize=30)
    ax.tick_params(labelsize=28)
    ax.set_xscale("log")
    ax.set_xticks([100, 500, 10000])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_ylim(0.0, 0.2)
    ax.set_yticks([0.0, 0.05, 0.10, 0.15])
    ax.legend(fontsize=12)
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    plt.tight_layout()


def recruitment_plot_mnist_between(mean_acc_dict, std_acc_dict, ns):
    """
    Plot the results from the recruitment experiment for MNIST -> fashion-MNIST.
    """
    # determine colors and labels for figure
    colors = sns.color_palette("Set1", n_colors=len(mean_acc_dict))
    labels = ["L2F (building)", "UF (new)", "recruiting", "hybrid"]

    # plot and format figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for i, key in enumerate(mean_acc_dict):
        ax.plot(ns, mean_acc_dict[key], c=colors[i], label=labels[i])
        upper_bound = np.array(mean_acc_dict[key]) + 1.96 * np.array(std_acc_dict[key])
        lower_bound = np.array(mean_acc_dict[key]) - 1.96 * np.array(std_acc_dict[key])
        ax.fill_between(
            ns,
            upper_bound,
            lower_bound,
            where=upper_bound >= lower_bound,
            facecolor=colors[i],
            alpha=0.15,
            interpolate=False,
        )
    ax.set_title("MNIST -> Fashion-MNIST Recruitment", fontsize=30)
    ax.set_ylabel("Generalization Error (Task 2)", fontsize=28)
    ax.set_xlabel("Number of Task 2 Samples", fontsize=30)
    ax.tick_params(labelsize=28)
    ax.set_xscale("log")
    ax.set_xticks([100, 500, 5000, 50000])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_ylim(0.10, 0.60)
    ax.set_yticks([0.15, 0.25, 0.35, 0.45, 0.55])
    ax.legend(fontsize=12)
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    plt.tight_layout()
