import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)

from base import BaseVoter

class ForestVoterClassification(BaseVoter):
    def __init__(self, sampled_indices_by_tree=None, finite_sample_corretion=False):
        """
        Doc strings here.
        """

        self.finite_sample_corretion = finite_sample_corretion
        self._is_fitted = False


    def fit(self, X, y):
        """
        Doc strings here.
        """
        self.tree_to_leaf_to_posterior = {}

        X, y = check_X_y(X, y)

        n, n_trees = X.shape

        for tree_id in range(n_trees):
            leaf_id_to_posterior = {}
            leaf_ids = np.unique(X[:, tree_id])
            for leaf_id in leaf_ids:
                X_in_leaf_id = np.where(X[:, tree_id] == leaf_id)[0]
                class_counts = [len(np.where(y[X_in_leaf_id] == y_)[0]) for y_ in np.unique(y)]
                posteriors = np.nan_to_num(np.array(class_counts) / np.sum(class_counts))

                if self.finite_sample_corretion:
                    posteriors = self._finite_sample_correction(posteriors, len(X_in_leaf_id), len(np.unique(y)))
                
                leaf_id_to_posterior[leaf_id] = posteriors
            self.tree_to_leaf_to_posterior[tree_id] = leaf_id_to_posterior

        self._is_fitted = True

        return self


    def vote(self, X):
        """
        Doc strings here.
        """

        if not self.is_fitted():
            msg = (
                    "This %(name)s instance is not fitted yet. Call 'fit' with "
                    "appropriate arguments before using this voter."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})
        
        X = check_array(X)

        n, n_trees = X.shape

        print(self.tree_to_leaf_to_posterior)

        posteriors = []
        for i, x in enumerate(X):
            posteriors.append(np.mean([self.tree_to_leaf_to_posterior[tree_id][x[tree_id]] for tree_id in range(n_trees)], axis=0))
        
        return np.array(posteriors)


    def is_fitted(self):
        """
        Doc strings here.
        """

        return self._is_fitted

    def _finite_sample_correction(posteriors, num_points_in_partition, num_classes):
        '''
        encourage posteriors to approach uniform when there is low data
        '''
        correction_constant = 1 / (num_classes * num_points_in_partition)

        zero_posterior_idxs = np.where(posteriors == 0)[0]
        posteriors[zero_posterior_idxs] = correction_constant
        
        posteriors /= sum(posteriors)
        
        return posteriors

class KNNVoterClassification(BaseVoter):
    def __init__(self, k, kwargs):
        """
        Doc strings here.
        """
        self._is_fitted = False
        self.k = k
        self.kwargs = kwargs
        
    def fit(self, 
            X, 
            y):
        """
        Doc strings here.
        """
        X, y = check_X_y(X, y)
        self.knn = KNeighborsClassifier(self.k, **self.kwargs)
        self.knn.fit(X, y)
        self._is_fitted = True
        
        return self
        
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