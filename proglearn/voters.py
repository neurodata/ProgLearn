"""
Main Author: Will LeVine 
Corresponding Email: levinewill@icloud.com
"""
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
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
    finite_sample_correction : bool
        boolean indicating whether this voter
        will have finite sample correction

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

    def __init__(self, finite_sample_correction=False, classes=[]):
        self.finite_sample_correction = finite_sample_correction
        self.classes = classes

    def fit(self, X, y, transformer):
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

        if np.asarray(self.classes).size != 0 and num_fit_classes < len(self.classes):
            for label in self.classes:
                if label not in np.unique(y):
                    self.missing_label_indices_.append(label)

        num_classes = num_fit_classes + len(self.missing_label_indices_)

        self.uniform_posterior_ = np.ones(num_classes) / num_classes

        self.leaf_to_posterior_ = {}
        
        self.transformer_ = transformer
        X = self.transformer_.transform(X)

        for leaf_id in np.unique(X):
            idxs_in_leaf = np.where(X == leaf_id)[0]
            class_counts = [
                len(np.where(y[idxs_in_leaf] == y_val)[0]) for y_val in np.unique(y)
            ]
            posteriors = np.nan_to_num(np.array(class_counts) / np.sum(class_counts))

            if self.finite_sample_correction:
                posteriors = self._finite_sample_correction(
                    posteriors, len(idxs_in_leaf), num_classes
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
        for x in self.transformer_.transform(X):
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
        return np.argmax(self.predict_proba(X), axis=1)

    def _finite_sample_correction(posteriors, num_points_in_partition, num_classes):
        """
        Encourage posteriors to approach uniform when there is low data through a finite sample correction.


        Parameters
        ----------
        posteriors : array of shape[n_samples, n_classes]
            posterior of each class for each sample
        num_points_in_partition : int
            number of samples in this particular transformation
        num_classes : int
            number of classes or labels

        Returns
        -------
        y_proba_hat : ndarray of shape [n_samples, n_classes]
            posteriors per example
        """
        correction_constant = 1 / (num_classes * num_points_in_partition)

        zero_posterior_idxs = np.where(posteriors == 0)[0]
        posteriors[zero_posterior_idxs] = correction_constant

        posteriors /= sum(posteriors)

        return posteriors
    
class KDTClassificationVoter(BaseClassificationVoter):
    def __init__(self, classes=[]):
        """
        Doc strings here.
        """
        self.classes = classes
        
    def fit(self, X, y, transformer):
        check_classification_targets(y)
        _, y = np.unique(y, return_inverse = True)
        
        num_fit_classes = len(np.unique(y))
        self.missing_label_indices_ = []

        if np.asarray(self.classes).size != 0 and num_fit_classes < len(self.classes):
            for label in self.classes:
                if label not in np.unique(y):
                    self.missing_label_indices_.append(label)
        
        polytope_ids = transformer.transform(X)
        
        #an array of all of the X-values of the class-wise means in each leaf
        self.polytope_means_X = []

        #an array of all of the y-values (i.e. class values) of the class-wise means in each leaf
        self.polytope_means_y = []

        #an array of all of the number of points that comprise the means in each leaf
        self.polytope_means_weight = []

        #an array of all of the average variances of the points in each leaf corresponding to 
        #a single class from the class-wise mean in that leaf
        self.polytope_means_cov = []
        
        for polytope_value in np.unique(polytope_ids, axis = 0):
            for y_val in np.unique(y):
                idxs_in_polytope_of_class = np.where((y == y_val) & (polytope_ids == polytope_value))[0]
                if len(idxs_in_polytope_of_class) > 1:
                    mean = np.mean(X[idxs_in_polytope_of_class], axis = 0)
                    self.polytope_means_X.append(mean)
                    #we already know the y value, so just append it 
                    self.polytope_means_y.append(y_val)
                    #compute the number of points in that leaf corresponding to that y value
                    #and append to the aggregate array
                    self.polytope_means_weight.append(len(idxs_in_polytope_of_class))
                    #compute the distances of all the points in that leaf corresponding to that y value
                    #from the mean X-value of the points in that leaf corresponding to that y value
                    #compute the covariance as the average distance of the class-wise points in that leaf 
                    #and append to the aggregate array
                    cov = np.mean((X[idxs_in_polytope_of_class] - mean) ** 2, axis = 0)
                    self.polytope_means_cov.append(np.eye(len(cov)) * cov)
        return self
                    
    def predict_proba(self, X, likelihood = True):
        check_is_fitted(self)
        y_vals = np.unique(self.polytope_means_y)
        votes_per_example = np.zeros((len(X), len(y_vals)))
        
        def compute_pdf(X, polytope_mean_id):
            polytope_mean_X = self.polytope_means_X[polytope_mean_id]
            polytope_mean_y = self.polytope_means_y[polytope_mean_id]
            polytope_mean_weight = self.polytope_means_weight[polytope_mean_id]
            covs_of_class = np.array(self.polytope_means_cov)[np.where(polytope_mean_y ==self.polytope_means_y)[0]]
            weights_of_class = np.array(self.polytope_means_weight)[np.where(polytope_mean_y ==self.polytope_means_y)[0]]
            polytope_mean_cov = np.average(covs_of_class, weights = weights_of_class, axis = 0)
            var = multivariate_normal(mean=polytope_mean_X, cov=polytope_mean_cov, allow_singular=True)
            return var.pdf(X) * polytope_mean_weight / np.sum(weights_of_class)
        
        likelihoods = np.array([compute_pdf(X, polytope_mean_id) for polytope_mean_id in range(len(self.polytope_means_X))])
        for polytope_id in range(len(likelihoods)):
            votes_per_example[:, self.polytope_means_y[polytope_id]] += likelihoods[polytope_id]
            
        if len(self.missing_label_indices_) > 0:
            for i in self.missing_label_indices_:
                new_col = np.zeros(votes_per_example.shape[0])
                votes_per_example = np.insert(votes_per_example, i, new_col, axis=1)
                
        if not likelihood:
            #normalize the posteriors per example so the posterior per example
            #sums to 1
            sum_per_example = np.sum(votes_per_example, axis = 1)
            for y_val in range(np.shape(votes_per_example)[1]):
                votes_per_example[:, y_val] /= sum_per_example
                                
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
        return np.argmax(self.predict_proba(X), axis=1)


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
        self.classes = classes

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

        if np.asarray(self.classes).size != 0 and num_classes < len(self.classes):
            for label in self.classes:
                if label not in np.unique(y):
                    self.missing_label_indices_.append(label)

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
        return np.argmax(self.predict_proba(X), axis=1)
