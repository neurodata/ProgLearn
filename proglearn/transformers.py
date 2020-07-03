import numpy as np
import abc

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)

class BaseTransformer(abc.ABC):
    """
    Doc strings here.
    """

    @abc.abstractmethod
    def fit(self, X, y=None):
        """
        Doc strings here.
        """

        pass

    @abc.abstractmethod
    def transform(self, X):
        """
        Doc strings here.
        """

        pass

    @abc.abstractmethod
    def is_fitted(self):
        """
        Doc strings here.
        """

        pass


class ForestTransformer(BaseTransformer):
    def __init__(self,
        bagger=BaggingClassifier,
        bagger_kwargs = {'n_estimators': 100, 'max_samples': 1.0, 'bootstrap': False},
        learner=DecisionTreeClassifier,
        learner_kwargs = {'max_depth':30, 'max_features': 'sqrt', 'min_samples_leaf': 1}
    ):
        """
        Doc strings here.
        """

        self.bagger = bagger
        self.bagger_kwargs = bagger_kwargs

        self.learner = learner
        self.learner_kwargs = learner_kwargs

        self._is_fitted = False


    def fit(self, X, y):
        """
        Doc strings here.
        """

        X, y = check_X_y(X, y)
        
        #define the ensemble
        self.transformer = self.bagger(
            self.learner(
                **self.learner_kwargs
            ),
            **self.bagger_kwargs
        )

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
        return np.array([tree.apply(X) for tree in self.transformer.estimators_]).T


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