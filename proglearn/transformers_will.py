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
        learner=DecisionTreeClassifier,
        max_depth=30,
        min_samples_leaf=1,
        max_samples = 0.63,
        max_features_tree = "auto",
        n_estimators=100,
        bootstrap=False,
    ):
        """
        Doc strings here.
        """

        self.bagger = bagger
        self.learner = learner
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_samples = max_samples
        self.max_features_tree = max_features_tree
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self._is_fitted = False


    def fit(self, X, y):
        """
        Doc strings here.
        """

        X, y = check_X_y(X, y)
        
        #define the ensemble
        self.ensemble = self.bagger(
            self.learner(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features_tree
            ),
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            bootstrap=self.bootstrap,
        )

        self.ensemble.fit(X, y)
        self._is_fitted = True


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
        return np.array([tree.apply(X) for tree in self.ensemble.estimators_])


    def is_fitted(self):
        """
        Doc strings here.
        """

        return self._is_fitted