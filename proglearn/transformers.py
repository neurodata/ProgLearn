"""
Main Author: Will LeVine 
Corresponding Email: levinewill@icloud.com
"""
import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)

from sklearn.utils.multiclass import check_classification_targets

import keras as keras

from .base import BaseTransformer


class NeuralClassificationTransformer(BaseTransformer):
    """
    A class used to transform data from a category to a specialized representation.

    Parameters
    ----------
    network : object
        A neural network used in the classification transformer.
    euclidean_layer_idx : int
        An integer to represent the final layer of the transformer.
    optimizer : str
        An optimizer used when compiling the neural network.
    loss : str, default="categorical_crossentropy"
        A loss function used when compiling the neural network.
    pretrained : bool, default=False
        A boolean used to identify if the network is pretrained.
    compile_kwargs : dict, default={"metrics": ["acc"]}
        A dictionary containing metrics for judging network performance.
    fit_kwargs : dict, default={
                "epochs": 100,
                "callbacks": [keras.callbacks.EarlyStopping(patience=5, monitor="val_acc")],
                "verbose": False,
                "validation_split": 0.33,
            },
        A dictionary to hold epochs, callbacks, verbose, and validation split for the network.

    Attributes (class)
    ----------
    None

    Attributes (object)
    ----------
    network : object
        A Keras model cloned from the network parameter.
    encoder : object
        A Keras model with inputs and outputs based on the network attribute. Output layers
        are determined by the euclidean_layer_idx parameter.
    _is_fitted : bool
        A boolean to identify if the network has already been fitted.
    optimizer : str
        A string to identify the optimizer used in the network.
    loss : str
        A string to identify the loss function used in the network.
    compile_kwargs : dict
        A dictionary containing metrics for judging network performance.
    fit_kwargs : dict
        A dictionary to hold epochs, callbacks, verbose, and validation split for the network.

    Methods
    ----------
    fit(X, y)
        Fits the transformer to data X with labels y.
    transform(X)
        Performs inference using the transformer.
    is_fitted()
        Indicates whether the transformer is fitted.
    """

    def __init__(
        self,
        network,
        euclidean_layer_idx,
        optimizer,
        loss="categorical_crossentropy",
        pretrained=False,
        compile_kwargs={"metrics": ["acc"]},
        fit_kwargs={
            "epochs": 100,
            "callbacks": [keras.callbacks.EarlyStopping(patience=5, monitor="val_acc")],
            "verbose": False,
            "validation_split": 0.33,
        },
    ):
        self.network = keras.models.clone_model(network)
        self.encoder = keras.models.Model(
            inputs=self.network.inputs,
            outputs=self.network.layers[euclidean_layer_idx].output,
        )
        self._is_fitted = pretrained
        self.optimizer = optimizer
        self.loss = loss
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs

    def fit(self, X, y):
        """
        Fits the transformer to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response data matrix).
        """

        check_classification_targets(y)
        _, y = np.unique(y, return_inverse=True)
        self.num_classes = len(np.unique(y))

        # more typechecking
        self.network.compile(
            loss=self.loss, optimizer=self.optimizer, **self.compile_kwargs
        )
        self.network.fit(
            X,
            keras.utils.to_categorical(y, num_classes=self.num_classes),
            **self.fit_kwargs
        )
        self._is_fitted = True

        return self

    def transform(self, X):
        """
        Performs inference using the transformer.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """

        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        # type check X
        return self.encoder.predict(X)

    def is_fitted(self):
        """
        Indicates whether the transformer is fitted.

        Parameters
        ----------
        None
        """

        return self._is_fitted


class TreeClassificationTransformer(BaseTransformer):
    """
    A class used to transform data from a category to a specialized representation.

    Attributes (object)
    ----------
    kwargs : dict
        A dictionary to contain parameters of the tree.
    _is_fitted_ : bool
        A boolean to identify if the model is currently fitted.

    Methods
    ----------
    fit(X, y)
        Fits the transformer to data X with labels y.
    transform(X)
        Performs inference using the transformer.
    is_fitted()
        Indicates whether the transformer is fitted.
    """

    def __init__(self, kwargs={}):

        self.kwargs = kwargs

        self._is_fitted = False

    def fit(self, X, y):
        """
        Fits the transformer to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response data matrix).
        """

        X, y = check_X_y(X, y)

        # define the ensemble
        self.transformer = DecisionTreeClassifier(**self.kwargs).fit(X, y)

        self._is_fitted = True

        return self

    def transform(self, X):
        """
        Performs inference using the transformer.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """

        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)
        return self.transformer.apply(X)

    def is_fitted(self):
        """
        Indicates whether the transformer is fitted.

        Parameters
        ----------
        None
        """

        return self._is_fitted
