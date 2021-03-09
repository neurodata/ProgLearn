"""
Main Author: Will LeVine
Corresponding Email: levinewill@icloud.com
"""
import warnings
import keras
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

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

    optimizer : str or keras.optimizers instance
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

    Attributes
    ----------
    encoder_ : object
        A Keras model with inputs and outputs based on the network attribute.
        Output layers are determined by the euclidean_layer_idx parameter.
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
        self.encoder_ = keras.models.Model(
            inputs=self.network.inputs,
            outputs=self.network.layers[euclidean_layer_idx].output,
        )
        self.pretrained = pretrained
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

        Returns
        -------
        self : NeuralClassificationTransformer
            The object itself.
        """
        check_X_y(X, y)
        _, y = np.unique(y, return_inverse=True)

        # more typechecking
        self.network.compile(
            loss=self.loss, optimizer=self.optimizer, **self.compile_kwargs
        )

        self.network.fit(X, keras.utils.to_categorical(y), **self.fit_kwargs)

        return self

    def transform(self, X):
        """
        Performs inference using the transformer.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        Returns
        -------
        X_transformed : ndarray
            The transformed input.

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        check_is_fitted(self)
        check_array(X)
        return self.encoder_.predict(X)


class TreeClassificationTransformer(BaseTransformer):
    """
    A class used to transform data from a category to a specialized representation.

    Parameters
    ----------
    max_features : {"auto", "sqrt", "log2"}, int or float, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    poisson_sampler : boolean, default=False
        To match the GRF theory [#1grf]_, if True, the number of features
        considered at each tree are drawn from a poisson distribution with
        mean equal to `max_features`.

    fit_kwargs : dict, default={}
        Named arguments passed to the sklearn.tree.DecisionTreeClassifier tree
        created during `fit`.

    Attributes
    ----------
    transformer_ : sklearn.tree.DecisionTreeClassifier
        an internal sklearn.tree.DecisionTreeClassifier.

    n_features_ : int
        The number of features of the data fitted.

    References
    ----------
    .. [#1grf] Athey, Susan, Julie Tibshirani and Stefan Wager.
    "Generalized Random Forests", Annals of Statistics, 2019.
    """

    def __init__(self, max_features=1.0, poisson_sampler=False, fit_kwargs={}):
        self.max_features = max_features
        self.poisson_sampler = poisson_sampler
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

        Returns
        -------
        self : TreeClassificationTransformer
            The object itself.
        """
        X, y = check_X_y(X, y)
        self.n_features_ = X.shape[1]
        if self.poisson_sampler:
            if self.max_features in ("auto", "sqrt"):
                max_features = np.sqrt(self.n_features_)
            elif self.max_features == "log2":
                max_features = np.log2(self.n_features_)
            elif isinstance(self.max_features, float):
                assert self.max_features > 0, self.max_features
                max_features = self.max_features * self.n_features_
            elif isinstance(self.max_features, int):
                assert self.max_features > 0, self.max_features
                max_features = self.max_features
            else:
                raise ValueError(f"max_features value not an accepted value")
            if max_features > self.n_features_:
                warnings.warn(
                    "max_features value led to poisson mean "
                    + "({max_features}) > the number of features"
                )
            max_features = int(max_features)
            max_features = min(max(np.random.poisson(max_features), 1), self.n_features_)
        else:
            max_features = self.max_features

        self.transformer_ = DecisionTreeClassifier(
            max_features=max_features, **self.fit_kwargs
        ).fit(X, y)
        return self

    def transform(self, X):
        """
        Performs inference using the transformer.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        Returns
        -------
        X_transformed : ndarray
            The transformed input.

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self.transformer_.apply(X)
