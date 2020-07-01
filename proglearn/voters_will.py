import numpy as np
import abc

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)

class BaseVoter(abc.ABC):
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
    def vote(self, X):
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
    
class KNNVoterClassification(BaseVoter):
    def __init__(self):
        """
        Doc strings here.
        """
        self._is_fitted = False
        
    def fit(self, 
            X, 
            y,
            k,
            kwargs):
        """
        Doc strings here.
        """
        X, y = check_X_y(X, y)
        self.knn = KNeighborsClassifier(k, **kwargs)
        self.knn.fit(X, y)
        self._is_fitted = True
        
    def vote(self, X):
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
        return self.knn.predict_proba(X)
    
    def is_fitted(self):
        """
        Doc strings here.
        """

        return self._is_fitted


class ForestVoterClassification(BaseVoter):
    def __init__(self):
        pass
        """
        Doc strings here.
        """

     #    self.bagger = bagger
     #    self.learner = learner
        # self.max_depth = max_depth
        # self.min_samples_leaf = min_samples_leaf
        # self.max_samples = max_samples
        # self.max_features_tree = max_features_tree
        # self.n_estimators = n_estimators
        # self.bootstrap = bootstrap

    def fit(self, X, y):
        """
        Doc strings here.
        """

        X, y = check_X_y(X, y)





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

class ForestVoterRegression(BaseVoter):
    pass
