import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import keras

__all__ = [
    "rf_classifier",
    "binary_dn",
    "sparse_parity",
    "test_suite",
    "plot_sample_size_experiment",
]


def rf_classifier(X_train, y_train, num_trees, max_depth, n_jobs, verbose):
    """
    Returns a random forest classifier model trained on data X_train and labels y_train

    Args:
        X_train : ndarray
            A numpy array of input data with shape [num_examples, num_features].
        y_train : ndarray
            A numpy array of data labels with shape [num_examples,].
        num_trees : int
            An int to identify number of trees to train the classifier.
        max_depth : int
            An int to identify the max depth of each tree in the classifier.

    Returns:
        rf_model : object
            A random forest classifier model for predicting new data.
    """

    rf_model = RandomForestClassifier(
        n_estimators=num_trees, max_depth=max_depth, n_jobs=n_jobs, verbose=verbose
    )
    rf_model.fit(X_train, y_train)

    return rf_model


def binary_dn(
    X_train,
    y_train,
    hidden_nodes,
    batch_size,
    epochs,
    learning_rate,
    validation_split,
    verbose,
):
    """
    Returns a binary neural network model trained on data X_train and labels y_train

    Args:
        X_train : ndarray
            A numpy array of input data with shape [num_examples, num_features].
        y_train : ndarray
            A numpy array of data labels with shape [num_examples,].
        hidden_nodes : int
            An int to identify the number of nodes in the hidden layer.
        batch_size : int
            An int to identify the batch size for training the model.
        epochs : int
            An int to identify the total number of epochs for training the model.
        learning_rate : float
            A float to identify the learning rate of the Adam optimizer.
        validation_split : float
            A float between 0 and 1 to represent the amount of training data to be
            used for validation.
        verbose : int
            An int to represent the level of output while training the model.

    Returns:
        dn_model : object
            A trained binary neural network.
    """

    dn_model = keras.Sequential()

    dn_model.add(keras.layers.Dense(X_train.shape[1], activation="relu"))
    dn_model.add(keras.layers.Dense(hidden_nodes, activation="relu"))
    dn_model.add(keras.layers.Dense(units=1, activation="sigmoid"))

    adam_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    dn_model.compile(
        optimizer=adam_optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )

    dn_model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
    )

    return dn_model


def sparse_parity(num_samples, p, p_star):
    """
    Generates simulation data from the sparse parity distribution of size num_samples
    using parameters p and p_star

    Args:
        num_samples : int
            An int corresponding to the number of samples generated with corresponding
            y label.
        p : int
            An int to identify the number of dimentions of the simulation data, X.
        p_star : int
            An int to identify the number of informative dimensions of the simulation
            data, X.

    Returns:
        X : ndarray
            A ndarray of size [num_samples, p] with each value uniformly distributed
            from [-1, 1].
        y : ndarray
            A ndarray of size [num_samples,] with data labels.
    """

    X = np.random.uniform(-1, 1, [num_samples, p])
    y = np.sum(np.where(X[:, :p_star] >= 0, 1, 0), axis=1) % 2

    return X, y


def test_suite(
    training_sample_sizes,
    testing_sample_size,
    trials,
    p,
    p_star,
    num_trees=500,
    max_depth=None,
    n_jobs=-1,
    rf_verbose=0,
    hidden_nodes=4,
    batch_size=3,
    epochs=10,
    learning_rate=0.001,
    validation_split=0.3,
    dn_verbose=0,
):
    """
    Runs a testing suite to evaluare RF and DN classifiers over a range of sample sizes
    and a given number of trials. Returns an ndarray for each classifier that
    contains individual model accuracies for each sample size and each trial.

    Args:
        training_sample_sizes : list
            A list containing sample sizes to train each model with. The model iterates
            over each sample size and trains a separate DN and RF model with the provided
            amount of data.
        testing_sample_size : int
            An int representing the size of the generated test set that will be used
            to evaluate each model. The test data is regenerated for each loop.
        trials : int
            An int representing the number of trials the model is run for. A higher
            trials number produces smoother outputs, but takes more time to run.
        p : int
            An int to identify the number of dimensions for generating the sparse
            parity data.
        p_star : int
            An int to identify the number of informative dimensions for generating
            the sparse parity data.
        num_trees : int, default=500
            An int to represent the number of trees for fitting the RF classifier.
        max_depth : int, default=None
            An int to represent the max depth for fitting the RF classifier.
        n_jobs : int, default=None
            An int to identify the number of jobs to run in parallel for the RF model.
            Increasing n_jobs uses additional processors to parallelize training across
            trees.
        rf_verbose : int, default=0
            An int to identify the output of the RF model while trained.
        hidden_nodes : int, default=4
            An int to identify the number of nodes in the hidden layer of the DN.
        batch_size : int, default=50
            An int to identity the batch size for training the DN model. For experimental
            parameters, this will need to be changed to use mini-batch sizes.
        epochs : int, default=10
            An int to represent the number of epochs over which the DN is trained.
        learning_rate : float, default=0.001
            A float to represent the learning rate of the Adam optimizer for training
            the DN.
        validation_split : float, default=0.3
            A float to represent the fraction of training data to be used as validation
            data.
        dn_verbose : int, default=0
            An int to identify the output of the DN model while trained.

    Returns:
        rf_evolution : ndarray
            A ndarray of size [len(sample_sizes), trials] containing the RF model
            accuracies across sample sizes and trials.

        dn_evolution : ndarray
            A ndarray of size [len(sample_sizes), trials] containing the DN model
            accuracies across sample sizes and trials.
    """

    rf_evolution = np.zeros((len(training_sample_sizes), trials))
    dn_evolution = np.zeros((len(training_sample_sizes), trials))

    for trial in range(trials):

        for sample_size_index, max_sample_size in enumerate(training_sample_sizes):

            X_train, y_train = sparse_parity(max_sample_size, p, p_star)
            X_train = X_train.astype("float32")
            y_train = y_train.astype("float32")

            X_test, y_test = sparse_parity(testing_sample_size, p, p_star)

            rf_model = rf_classifier(
                X_train=X_train,
                y_train=y_train,
                num_trees=num_trees,
                max_depth=max_depth,
                n_jobs=n_jobs,
                verbose=rf_verbose,
            )

            rf_predictions = rf_model.predict(X_test)
            rf_error = 1 - accuracy_score(y_test, rf_predictions)
            rf_evolution[sample_size_index][trial] = rf_error

            dn_model = binary_dn(
                X_train=X_train,
                y_train=y_train,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_split=validation_split,
                hidden_nodes=hidden_nodes,
                verbose=dn_verbose,
            )

            score = dn_model.evaluate(
                X_test,
                y_test,
                batch_size=None,
                verbose=0,
                sample_weight=None,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                return_dict=False,
            )

            dn_error = 1 - score[1]
            dn_evolution[sample_size_index][trial] = dn_error

    return rf_evolution, dn_evolution


def plot_sample_size_experiment(
    rf_evolution, dn_evolution, training_sample_sizes, p, p_star
):
    """
    Plots a figure to visualize the evolution of the RF model and the DN model over
    the range of sample sizes. Error bars are calculated based on the standard
    error of the mean.

    Args:
        rf_evolution : ndarray
            A ndarray of size [len(sample_sizes), trials] containing the RF model
            accuracies across sample sizes and trials.
        dn_evolution : ndarray
            A ndarray of size [len(sample_sizes), trials] containing the DN model
            accuracies across sample sizes and trials.
        training_sample_sizes : list
            A list containing the sample sizes the model was trained with.
        p : int
            The p value for the sparse parity distribution. This is used for plotting
            the title of the figure.
        p_star : int
            The p_star value for the sparse parity distribution. This is used for
            plotting the title of the figure.

    Returns:
        None
    """
    dn_evolution_mean = np.mean(dn_evolution, axis=1)
    rf_evolution_mean = np.mean(rf_evolution, axis=1)

    yerr_dn = stats.sem(dn_evolution, axis=1)
    yerr_rf = stats.sem(rf_evolution, axis=1)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    ax.errorbar(
        training_sample_sizes,
        dn_evolution_mean,
        yerr=yerr_dn,
        linewidth=4,
        c="b",
        alpha=0.8,
        label="DN",
    )
    ax.errorbar(
        training_sample_sizes,
        rf_evolution_mean,
        yerr=yerr_rf,
        linewidth=5,
        c="r",
        alpha=0.8,
        label="RF",
    )

    plt.xticks(range(0, 25000, 5000), fontsize=20)
    plt.yticks(fontsize=20)

    plt.title("sparse parity: p={}, p*={}".format(p, p_star), fontsize=24)
    plt.xlabel("number of training samples", fontsize=24)
    plt.ylabel("classification error", fontsize=24)

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, prop={"size": 24})
