import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)

from sklearn.utils.multiclass import check_classification_targets

import keras as keras

from base import BaseTransformer

class NeuralTransformer(BaseTransformer):
    def __init__(self, 
                 network, 
                 euclidean_layer_idx, 
                 pretrained = False,
                 optimizer = keras.optimizers.Adam(3e-4),
                 loss = "categorical_crossentropy",
                 metrics = ['acc'],
                 epochs = 100,
                 callbacks = [keras.callbacks.EarlyStopping(patience = 5, monitor = "val_acc")],
                 verbose = False,
                 validation_split = .33,
                 compile_kwargs = {},
                 fit_kwargs = {}):
        """
        Doc strings here.
        """
        self.network = network
        self.encoder = keras.models.Model(inputs = self.network.inputs, outputs = self.network.layers[euclidean_layer_idx].output)
        self.transformer_fitted_ = pretrained
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs 
        self.callbacks = callbacks
        self.verbose = verbose 
        self.validation_split = validation_split 
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
                             metrics = self.metrics, 
                             optimizer = self.optimizer, 
                             **self.compile_kwargs)
        self.network.fit(X, 
                    keras.utils.to_categorical(y), 
                    epochs = self.epochs, 
                    callbacks = self.callbacks, 
                    verbose = self.verbose,
                    validation_split = self.validation_split,
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

class TreeTransformer(BaseTransformer):
    def __init__(self,
        learner=DecisionTreeClassifier,
        learner_kwargs = {'max_depth':30, 'max_features': 'sqrt', 'min_samples_leaf': 1}
    ):
        """
        Doc strings here.
        """

        self.learner = learner
        self.learner_kwargs = learner_kwargs

        self._is_fitted = False


    def fit(self, X, y):
        """
        Doc strings here.
        """

        X, y = check_X_y(X, y)
        
        #define the ensemble
        self.transformer = self.learner(**self.learner_kwargs)

        self.transformer.fit(X, y)
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
        return tree.apply(X)


    def is_fitted(self):
        """
        Doc strings here.
        """

        return self._is_fitted