"""
Main Author: Will LeVine 
Corresponding Email: levinewill@icloud.com
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)
from sklearn.utils.multiclass import check_classification_targets
from .base import BaseClassificationVoter


class TreeClassificationVoter(BaseClassificationVoter):
    """
    A class used to vote on data transformed under a tree, which inherits from
    the BaseClassificationVoter class in base.py.

    Parameters
    ----------
    kappa : float
        coefficient for finite sample correction
        If set to default, no finite sample correction is performed.

    classes : list, default=[]
        list of all possible output label values

    Attributes
    ----------
    missing_label_indices_ : list
        a (potentially empty) list of label values
        that exist in the ``classes`` parameter but
        are missing in the latest ``fit`` function
        call

    uniform_posterior_ : ndarray of shape (n_classes,)
        the uniform posterior associated with the
    """

    def __init__(self, kappa=np.inf, classes=[]):
        self.kappa = kappa
        self.classes = np.asarray(classes)

    def fit(self, X, y):
        """
        Fits transformed data X given corresponding class labels y.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            the transformed input data
        y : array of shape [n_samples]
            the class labels

        Returns
        -------
        self : TreeClassificationVoter
            The object itself.
        """
        check_classification_targets(y)

        num_fit_classes = len(np.unique(y))
        self.missing_label_indices_ = []

        if self.classes.size != 0 and num_fit_classes < len(self.classes):
            for idx, label in enumerate(self.classes):
                if label not in np.unique(y):
                    self.missing_label_indices_.append(idx)

        self.uniform_posterior_ = np.ones(num_fit_classes) / num_fit_classes

        self.leaf_to_posterior_ = {}

        for leaf_id in np.unique(X):
            idxs_in_leaf = np.where(X == leaf_id)[0]
            class_counts = [
                len(np.where(y[idxs_in_leaf] == y_val)[0]) for y_val in np.unique(y)
            ]
            posteriors = np.nan_to_num(np.array(class_counts) / np.sum(class_counts))
            posteriors = self._finite_sample_correction(
                posteriors, len(idxs_in_leaf), self.kappa
            )
            self.leaf_to_posterior_[leaf_id] = posteriors

        return self

    def predict_proba(self, X):
        """
        Returns the posterior probabilities of each class for data X.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            the transformed input data

        Returns
        -------
        y_proba_hat : ndarray of shape [n_samples, n_classes]
            posteriors per example

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        check_is_fitted(self)
        votes_per_example = []
        for x in X:
            if x in list(self.leaf_to_posterior_.keys()):
                votes_per_example.append(self.leaf_to_posterior_[x])
            else:
                votes_per_example.append(self.uniform_posterior_)

        votes_per_example = np.array(votes_per_example)

        if len(self.missing_label_indices_) > 0:
            for i in self.missing_label_indices_:
                new_col = np.zeros(votes_per_example.shape[0])
                votes_per_example = np.insert(votes_per_example, i, new_col, axis=1)

        return votes_per_example

    def predict(self, X):
        """
        Returns the predicted class labels for data X.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            the transformed input data

        Returns
        -------
        y_hat : ndarray of shape [n_samples]
            predicted class label per example

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]

    def _finite_sample_correction(self, posteriors, num_points_in_partition, kappa):
        """
        Encourage posteriors to approach uniform when there is low data through a finite sample correction.


        Parameters
        ----------
        posteriors : array of shape[n_samples, n_classes]
            posterior of each class for each sample
        num_points_in_partition : int
            number of samples in this particular transformation
        kappa : float
            coefficient for finite sample correction

        Returns
        -------
        y_proba_hat : ndarray of shape [n_samples, n_classes]
            posteriors per example
        """
        correction_constant = 1 / (kappa * num_points_in_partition)

        zero_posterior_idxs = np.where(posteriors == 0)[0]
        posteriors[zero_posterior_idxs] = correction_constant

        posteriors /= sum(posteriors)

        return posteriors


class KNNClassificationVoter(BaseClassificationVoter):
    """
    A class used to vote on data under any transformer outputting data
    in continuous Euclidean space, which inherits from the BaseClassificationVoter
    class in base.py.

    Parameters
    ----------
    k : int
        integer indicating number of neighbors to use for each prediction during
        fitting and voting

    kwargs : dictionary, default={}
        contains all keyword arguments for the underlying KNN

    classes : list, default=[]
        list of all possible output label values

    Attributes
    ----------
    missing_label_indices_ : list
        a (potentially empty) list of label values
        that exist in the ``classes`` parameter but
        are missing in the latest ``fit`` function
        call

    knn_ : sklearn.neighbors.KNeighborsClassifier
        the internal sklearn instance of KNN
        classifier
    """

    def __init__(self, k=None, kwargs={}, classes=[]):
        self.k = k
        self.kwargs = kwargs
        self.classes = np.asarray(classes)

    def fit(self, X, y):
        """
        Fits data X given class labels y.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            the transformed data that will be trained on
        y : array of shape [n_samples]
            the label for class membership of the given data

        Returns
        -------
        self : KNNClassificationVoter
            The object itself.
        """
        X, y = check_X_y(X, y)
        k = int(np.log2(len(X))) if self.k == None else self.k
        self.knn_ = KNeighborsClassifier(k, **self.kwargs)
        self.knn_.fit(X, y)

        num_classes = len(np.unique(y))
        self.missing_label_indices_ = []

        if self.classes.size != 0 and num_classes < len(self.classes):
            for idx, label in enumerate(self.classes):
                if label not in np.unique(y):
                    self.missing_label_indices_.append(idx)

        return self

    def predict_proba(self, X):
        """
        Returns the posterior probabilities of each class for data X.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            the transformed input data

        Returns
        -------
        y_proba_hat : ndarray of shape [n_samples, n_classes]
            posteriors per example

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        check_is_fitted(self)
        X = check_array(X)
        votes_per_example = self.knn_.predict_proba(X)

        if len(self.missing_label_indices_) > 0:
            for i in self.missing_label_indices_:
                new_col = np.zeros(votes_per_example.shape[0])
                votes_per_example = np.insert(votes_per_example, i, new_col, axis=1)

        return votes_per_example

    def predict(self, X):
        """
        Returns the predicted class labels for data X.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            the transformed input data

        Returns
        -------
        y_hat : ndarray of shape [n_samples]
            predicted class label per example

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]
