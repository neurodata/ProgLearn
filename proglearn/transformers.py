"""
Main Author: Will LeVine
Corresponding Email: levinewill@icloud.com
"""
from tensorflow import keras
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

    fitted_ : boolean
        A boolean flag initialized after the model is fitted.
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
        check_X_y(X, y, ensure_2d=False, allow_nd=True)
        _, y = np.unique(y, return_inverse=True)

        self.network.compile(
            loss=self.loss, optimizer=self.optimizer, **self.compile_kwargs
        )

        self.network.fit(X, keras.utils.to_categorical(y), **self.fit_kwargs)
        self.fitted_ = True

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
        check_array(X, ensure_2d=False, allow_nd=True)
        check_is_fitted(self, attributes="fitted_")
        return self.encoder_.predict(X)


class TreeClassificationTransformer(BaseTransformer):
    """
    A class used to transform data from a category to a specialized representation.

    Parameters
    ----------
    kwargs : dict, default={}
        A dictionary to contain parameters of the tree.

    Attributes
    ----------
    transformer : sklearn.tree.DecisionTreeClassifier
        an internal sklearn DecisionTreeClassifier
    """

    def __init__(self, kwargs={}):
        self.kwargs = kwargs


    def _partial_fit(transformer,test, X, y, inputclasses):
        # print("in partial fit")
        # print(transformer)
        # print(test)
        
        X, y = check_X_y(X, y)
        test.partial_fit(X, y, classes = [0,1]) # set to [0,1] for testing
        #print(self)
        #print("leaving partial fit")
        return test

    # def _partial_fit(up_transformer, X, y, inputclasses=None):
    #     X, y = check_X_y(X, y)
    #     #self.transformer_ = DecisionTreeClassifier(**self.kwargs).partial_fit(X, y, classes = inputclasses)
    #     up_transformer.partial_fit(X, y, classes = inputclasses)
    #     return up_transformer

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
        self.transformer_ = DecisionTreeClassifier(**self.kwargs).fit(X, y)
        print(self.transformer_)
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
        X = check_array(X)
        check_is_fitted(self)
        return self.transformer_.apply(X)
