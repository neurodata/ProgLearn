'''
Main Author: Will LeVine 
Corresponding Email: levinewill@icloud.com
'''
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)

from sklearn.utils.multiclass import check_classification_targets

from .base import BaseVoter


class TreeClassificationVoter(BaseVoter):
    """
    A class used to vote on data transformed under a tree.

    Attributes
    ---
    finite_sample_correction : bool
        boolean indicating whether this voter
        will have finite sample correction

    Methods
    ---
    fit(X, y)
        fits tree classification to transformed data X with labels y
    vote(X)
        predicts posterior probabilities given transformed data, X, for each class
    is_fitted()
        returns if the classifier has been fitted for this transformation yet
    _finite_sample_correction(posteriors, num_points_in_partition, num_classes)
        performs finite sample correction on input data
    """
    def __init__(self, finite_sample_correction=False, classes=[]):
        self.finite_sample_correction = finite_sample_correction
        self._is_fitted = False
        self.classes = classes

    def fit(self, X, y):
        """
        Fits transformed data X given corresponding class labels y.

        Attributes
        ---
        X : array of shape [n_samples, n_features]
            the transformed input data
        y : array of shape [n_samples]
            the class labels
        """
        check_classification_targets(y)

        num_classes = len(np.unique(y))
        self.missing_label_indices = []

        if np.asarray(self.classes).size != 0 and num_classes < len(self.classes):
            for label in self.classes:
                if label not in np.unique(y):
                    self.missing_label_indices.append(label)        

        self.uniform_posterior = np.ones(num_classes) / num_classes

        self.leaf_to_posterior = {}

        for leaf_id in np.unique(X):
            idxs_in_leaf = np.where(X == leaf_id)[0]
            class_counts = [
                len(np.where(y[idxs_in_leaf] == y_val)[0]) for y_val in np.unique(y)
            ]
            posteriors = np.nan_to_num(np.array(class_counts) / np.sum(class_counts))

            if self.finite_sample_correction:
                posteriors = self._finite_sample_correction(
                    posteriors, len(idxs_in_leaf), len(np.unique(y))
                )

            self.leaf_to_posterior[leaf_id] = posteriors

        self._is_fitted = True

        return self

    def vote(self, X):
        """
        Returns the posterior probabilities of each class for data X.

        Attributes
        ---
        X : array of shape [n_samples, n_features]
            the transformed input data

        Raises
        ---
        NotFittedError :
            when the model has not yet been fit for this transformation
        """
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this voter."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        votes_per_example = []
        for x in X:
            if x in list(self.leaf_to_posterior.keys()):
                votes_per_example.append(self.leaf_to_posterior[x])
            else:
                votes_per_example.append(self.uniform_posterior)
        
        votes_per_example = np.array(votes_per_example)

        if len(self.missing_label_indices) > 0:
            for i in self.missing_label_indices:
                new_col = np.zeros(votes_per_example.shape[0])
                votes_per_example = np.insert(votes_per_example, i, new_col, axis=1)

        return votes_per_example

    def is_fitted(self):
        """
        Returns boolean indicating whether the voter has been fit.
        """

        return self._is_fitted

    def _finite_sample_correction(posteriors, num_points_in_partition, num_classes):
        """
        Encourage posteriors to approach uniform when there is low data through a finite sample correction.

        Attributes
        ---
        posteriors : array of shape[n_samples, n_classes]
            posterior of each class for each sample
        num_points_in_partition : int
            number of samples in this particular transformation
        num_classes : int
            number of classes or labels
        """
        correction_constant = 1 / (num_classes * num_points_in_partition)

        zero_posterior_idxs = np.where(posteriors == 0)[0]
        posteriors[zero_posterior_idxs] = correction_constant

        posteriors /= sum(posteriors)

        return posteriors


class KNNClassificationVoter(BaseVoter):
    """
    A class used to vote on data under any transformer 
    outputting data in continuous Euclidean space.

    Attributes
    ---
    k : int
        integer indicating number of neighbors to use for each prediction during 
        fitting and voting
    kwargs : dictionary
        contains all keyword arguments for the underlying KNN

    Methods
    ---
    fit(X, y)
        fits tree classification to transformed data X with labels y
    vote(X)
        predicts posterior probabilities given transformed data, X, for each class label
    is_fitted()
        returns if the classifier has been fitted for this transformation yet
    """
    def __init__(self, k=None, kwargs={}, classes=[]):
        self._is_fitted = False
        self.k = k
        self.kwargs = kwargs
        self.classes = classes

    def fit(self, X, y):
        """
        Fits data X given class labels y.

        Attributes
        ---
        X : array of shape [n_samples, n_features]
            the transformed data that will be trained on
        y : array of shape [n_samples]
            the label for class membership of the given data
        """
        X, y = check_X_y(X, y)
        self.k = int(np.log2(len(X))) if self.k == None else self.k
        self.knn = KNeighborsClassifier(self.k, **self.kwargs)
        self.knn.fit(X, y)
        self._is_fitted = True

        num_classes = len(np.unique(y))
        self.missing_label_indices = []

        if np.asarray(self.classes).size != 0 and num_classes < len(self.classes):
            for label in self.classes:
                if label not in np.unique(y):
                    self.missing_label_indices.append(label)

        return self

    def vote(self, X):
        """
        Returns the posterior probabilities of each class for data X.

        Attributes
        ---
        X : array of shape [n_samples, n_features]
            the transformed input data

        Raises
        ---
        NotFittedError :
            when the model has not yet been fit for this transformation
        """
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)
        votes_per_example = self.knn.predict_proba(X)

        if len(self.missing_label_indices) > 0:
            for i in self.missing_label_indices:
                new_col = np.zeros(votes_per_example.shape[0])
                votes_per_example = np.insert(votes_per_example, i, new_col, axis=1)
        return votes_per_example

    def is_fitted(self):
        """
        Returns boolean indicating whether the voter has been fit.
        """
        return self._is_fitted
