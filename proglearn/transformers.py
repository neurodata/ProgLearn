import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)

from sklearn.utils.multiclass import check_classification_targets

import keras as keras

from base import BaseTransformer

class NeuralClassificationTransformer(BaseTransformer):
    def __init__(self, 
                 network, 
                 euclidean_layer_idx, 
                 optimizer,
                 loss = "categorical_crossentropy",
                 pretrained = False,
                 num_classes = None,
                 compile_kwargs = {"metrics" : ['acc']},
                 fit_kwargs = {"epochs" : 100, 
                               "callbacks" : [keras.callbacks.EarlyStopping(patience = 5, monitor = "val_acc")],
                               "verbose" : False,
                               "validation_split" : .33
                              }):
        """
        Doc strings here.
        """
        self.network = keras.models.clone_model(network)
        self.encoder = keras.models.Model(inputs = self.network.inputs, 
                                          outputs = self.network.layers[euclidean_layer_idx].output
                                         )
        self._is_fitted = pretrained
        self.optimizer = optimizer
        self.loss = loss
        self.num_classes = num_classes
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs


    def fit(self, 
            X, 
            y):
        """
        Doc strings here.
        """
        check_classification_targets(y)
        _, y = np.unique(y, return_inverse=True)
        
        #more typechecking
        self.network.compile(loss = self.loss, 
                             optimizer = self.optimizer, 
                             **self.compile_kwargs)
        self.network.fit(X, 
                    keras.utils.to_categorical(y, num_classes = self.num_classes), 
                    **self.fit_kwargs)
        self._is_fitted = True
        
        return self


    def transform(self, X):
        """
        Doc strings here.
        """

        if not self.is_fitted():
            msg = (
                    "This %(name)s instance is not fitted yet. Call 'fit' with "
                    "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})
            
        #type check X
        return self.encoder.predict(X)


    def is_fitted(self):
        """
        Doc strings here.
        """

        return self._is_fitted

class TreeClassificationTransformer(BaseTransformer):
    def __init__(self, kwargs = {}
    ):
        """
        Doc strings here.
        """

        self.kwargs = kwargs

        self._is_fitted = False


    def fit(self, X, y):
        """
        Doc strings here.
        """

        X, y = check_X_y(X, y)
        
        #define the ensemble
        self.transformer = DecisionTreeClassifier(**self.kwargs).fit(X, y)

        self._is_fitted = True

        return self


    def transform(self, X):
        """
        Doc strings here.
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
        Doc strings here.
        """

        return self._is_fitted