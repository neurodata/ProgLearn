import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import time

__all__ = [
    "random_forest_classifier_model",
    "binary_deep_neural_network"
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
        n_estimators=num_trees, max_depth=max_depth,  n_jobs = n_jobs, verbose=verbose
    )
    rf_model.fit(X_train, y_train)

    return rf_model


def binary_dn(
    X_train, y_train, hidden_nodes, batch_size, epochs, learning_rate, validation_split, verbose
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
        dnn_model : object
            A trained binary neural network.
    """

    dnn_model = Sequential()

    dnn_model.add(Dense(X_train.shape[1], activation="relu"))
    dnn_model.add(Dense(hidden_nodes, activation="relu"))
    dnn_model.add(Dense(units=1, activation="sigmoid"))

    adam_optimizer = Adam(learning_rate=learning_rate)
    dnn_model.compile(
        optimizer=adam_optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )

    dnn_model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
    )

    return dnn_model


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
    sample_sizes,
    iterations,
    p,
    p_star,
    num_trees=500,
    max_depth=None,
    n_jobs=None,
    rf_verbose=0,
    hidden_nodes=4,
    batch_size=50,
    epochs=10,
    learning_rate=0.001,
    validation_split=0.3,
    dnn_verbose=0,
):
    """
    Runs a testing suite to evaluare RF and DN classifiers over a range of sample sizes 
    and a given number of iterations. Returns an ndarray for each classifier that 
    contains individual model accuracies for each sample size and each iteration.

    Args:
        sample_sizes : list
            A list containing sample sizes to train/test each model with. The sample 
            size values represent the total sparse parity data generated. To get the 
            individual training set sample size and test set sample size, these values 
            are fed through a train_test_split.
        iterations : int
            An int representing the number of iterations the model is run for. A higher 
            iterations number produces smoother outputs, but takes more time to run.
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
        dnn_verbose : int, default=0
            An int to identify the output of the DN model while trained.

    Returns:
        rf_evolution : ndarray
            A ndarray of size [len(sample_sizes), iterations] containing the RF model 
            accuracies across sample sizes and iterations.
            
        dnn_evolution : ndarray
            A ndarray of size [len(sample_sizes), iterations] containing the DNN model 
            accuracies across sample sizes and iterations.
    """

    rf_evolution = np.zeros((len(sample_sizes), iterations))
    dnn_evolution = np.zeros((len(sample_sizes), iterations))

    for iteration in range(iterations):

        for sample_size_index, max_sample_size in enumerate(sample_sizes):

            X, y = sparse_parity(max_sample_size, p, p_star)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=101
            )

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
            rf_evolution[sample_size_index][iteration] = rf_error

            dnn_model = binary_dn(
                X_train=X_train,
                y_train=y_train,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_split=validation_split,
                hidden_nodes=hidden_nodes,
                verbose=dnn_verbose,
            )
            
            score = dnn_model.evaluate(
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

            dnn_error = 1 - score[1]
            dnn_evolution[sample_size_index][iteration] = dnn_error

    return rf_evolution, dnn_evolution