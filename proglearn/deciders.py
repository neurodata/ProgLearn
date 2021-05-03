"""
Main Author: Will LeVine
Corresponding Email: levinewill@icloud.com
"""
import numpy as np

from .base import BaseClassificationDecider, BaseDecider

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


class SimpleAverage(BaseDecider):
    """
    A class for a decider that uses the average vote for regression.

    Parameters
    ----------
    classes : list, default=[]
        List of final output classification labels of type obj.

    Attributes
    ----------
    transformer_id_to_transformers_ : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a transformer. This dictionary
        maps transformers to a particular transformer id.

    transformer_id_to_voters_ : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a voter class. This dictionary
        maps voter classes to a particular transformer id.
    """

    def __init__(self, classes=[]):
        self.classes = classes

    def fit(
        self,
        X,
        y,
        transformer_id_to_transformers,
        transformer_id_to_voters,
        classes=None,
    ):
        """
        Function for fitting.
        Stores attributes (classes, transformer_id_to_transformers,
        and transformer_id_to_voters) of a BaseDecider.

        Parameters:
        -----------
        X : ndarray
            Input data matrix.

        y : ndarray
            Output (i.e. response) data matrix.

        transformer_id_to_transformers : dict
            A dictionary with keys of type obj corresponding to transformer ids
            and values of type obj corresponding to a transformer. This dictionary
            maps transformers to a particular transformer id.

        transformer_id_to_voters : dict
            A dictionary with keys of type obj corresponding to transformer ids
            and values of type obj corresponding to a voter class. This dictionary thus
            maps voter classes to a particular transformer id.

        classes : list, default=None
            List of final output classification labels of type obj.

        Returns
        -------
        self : SimpleAverage
            The object itself.
        """
        self.classes = self.classes if len(self.classes) > 0 else np.unique(y)
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters

        return self

    def predict_proba(self, X, transformer_ids=None):
        """
        Predicts posterior probabilities per input example.

        Loops through each transformer and bag of transformers.
        Performs a transformation of the input data with the transformer.
        Gets a voter to map the transformed input data into a posterior distribution.
        Gets the mean vote per bagging component and append it to a vote per transformer id.
        Returns the aggregate average vote.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        transformer_ids : list, default=None
            A list with specific transformer ids that will be used for inference. Defaults
            to using all transformers if no transformer ids are given.

        Returns
        -------
        y_proba_hat : ndarray of shape [n_samples, n_classes]
            posteriors per example
        """
        vote_per_transformer_id = []
        for transformer_id in (
            transformer_ids
            if transformer_ids is not None
            else self.transformer_id_to_voters.keys()
        ):
            vote_per_bag_id = []
            for bag_id in range(
                len(self.transformer_id_to_transformers[transformer_id])
            ):
                transformer = self.transformer_id_to_transformers[transformer_id][
                    bag_id
                ]
                X_transformed = transformer.transform(X)
                voter = self.transformer_id_to_voters[transformer_id][bag_id]
                vote = voter.predict(X_transformed)
                vote_per_bag_id.append(vote)
            vote_per_transformer_id.append(np.mean(vote_per_bag_id, axis=0))
        return np.mean(vote_per_transformer_id, axis=0)

    def predict(self, X, transformer_ids=None):
        """
        Predicts by calling predict_proba.

        Uses the predict_proba method to get the mean vote.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        transformer_ids : list, default=None
            A list with all transformer ids. Defaults to None if no transformer ids
            are given.

        Returns
        -------
        y_hat : ndarray of shape [n_samples]
            predicted class label per example
        """
        vote_overall = self.predict_proba(X, transformer_ids=transformer_ids)

        return vote_overall


class SimpleArgmaxAverage(BaseClassificationDecider):
    """
    A class for a decider that uses the average vote for classification.

    Parameters
    ----------
    classes : list, default=[]
        List of final output classification labels of type obj.

    Attributes
    ----------
    transformer_id_to_transformers_ : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a transformer. This dictionary
        maps transformers to a particular transformer id.

    transformer_id_to_voters_ : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a voter class. This dictionary
        maps voter classes to a particular transformer id.
    """

    def __init__(self, classes=[]):
        self.classes = classes

    def fit(
        self, X, y, transformer_id_to_transformers, transformer_id_to_voters,
    ):
        """
        Function for fitting.
        Stores attributes (classes, transformer_id_to_transformers,
        and transformer_id_to_voters) of a ClassificationDecider.

        Parameters:
        -----------
        X : ndarray
            Input data matrix.

        y : ndarray
            Output (i.e. response) data matrix.

        transformer_id_to_transformers : dict
            A dictionary with keys of type obj corresponding to transformer ids
            and values of type obj corresponding to a transformer. This dictionary
            maps transformers to a particular transformer id.

        transformer_id_to_voters : dict
            A dictionary with keys of type obj corresponding to transformer ids
            and values of type obj corresponding to a voter class. This dictionary thus
            maps voter classes to a particular transformer id.

        Returns
        -------
        self : SimpleArgmaxAverage
            The object itself.

        Raises
        -------
        ValueError
            When the labels have not been provided and the classes are empty.
        """
        if not isinstance(self.classes, (list, np.ndarray)):
            if len(y) == 0:
                raise ValueError(
                    "Classification Decider classes undefined with no class labels fed to fit"
                )
            else:
                self.classes = np.unique(y)
        else:
            self.classes = np.array(self.classes)
        self.transformer_id_to_transformers_ = transformer_id_to_transformers
        self.transformer_id_to_voters_ = transformer_id_to_voters
        return self

    def predict_proba(self, X, transformer_ids=None):
        """
        Predicts posterior probabilities per input example.

        Loops through each transformer and bag of transformers.
        Performs a transformation of the input data with the transformer.
        Gets a voter to map the transformed input data into a posterior distribution.
        Gets the mean vote per bagging component and append it to a vote per transformer id.
        Returns the aggregate average vote.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        transformer_ids : list, default=None
            A list with specific transformer ids that will be used for inference. Defaults
            to using all transformers if no transformer ids are given.

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
        vote_per_transformer_id = []
        for transformer_id in (
            transformer_ids
            if transformer_ids is not None
            else self.transformer_id_to_voters_.keys()
        ):
            check_is_fitted(self)
            vote_per_bag_id = []
            for bag_id in range(
                len(self.transformer_id_to_transformers_[transformer_id])
            ):
                transformer = self.transformer_id_to_transformers_[transformer_id][
                    bag_id
                ]
                X_transformed = transformer.transform(X)
                voter = self.transformer_id_to_voters_[transformer_id][bag_id]
                vote = voter.predict_proba(X_transformed)
                vote_per_bag_id.append(vote)
            vote_per_transformer_id.append(np.mean(vote_per_bag_id, axis=0))
        return np.mean(vote_per_transformer_id, axis=0)

    def predict(self, X, transformer_ids=None):
        """
        Predicts the most likely class per input example.

        Uses the predict_proba method to get the mean vote per id.
        Returns the class with the highest vote.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        transformer_ids : list, default=None
            A list with all transformer ids. Defaults to None if no transformer ids
            are given.

        Returns
        -------
        y_hat : ndarray of shape [n_samples]
            predicted class label per example

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        vote_overall = self.predict_proba(X, transformer_ids=transformer_ids)
        return self.classes[np.argmax(vote_overall, axis=1)]


class KNNClassificationDecider(BaseClassificationDecider):
    """
    A class for a decider that uses a k-nearest neighbor (KNN) vote for classification.
    Normal implementation uses voter posteriors for all transformers for each bag, but
    creates class specific KNNs for each bag. These class specific KNNs predict a
    one-hot encoded output for their specific classes.
    
    Parameters
    ----------
    k : int, default=None
        Number of neighbors for KNN voting.

    classes : list, default=[]
        List of final output classification labels of type obj.

    Attributes
    ----------
    _is_fitted : boolean, default=False
        Parameter to keep track of whether the model is fitted.

    transformer_ids : list
        A list of transformer ids.

    transformer_id_to_transformers_ : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a transformer. This dictionary
        maps transformers to a particular transformer id.

    transformer_id_to_voters_ : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a voter class. This dictionary
        maps voter classes to a particular transformer id.
        
    knn : list, default=[]
        A list of sklearn.KNNClassifiers with length equal to the number of bags.
    """

    def __init__(self, k=None, classes=[]):
        self.k = k
        self._is_fitted = False
        self.classes = classes

    def fit(self, X, y, transformer_id_to_transformers, transformer_id_to_voters):
        """
        Fits a KNNClassifier for each bag.

        Get the voter posteriors from ensemble_representations.
        Loop through each bag.
        Fit a KNNClassifier to the voter posteriors along all transformers for each bag.
        Repeat this for each class to get class-specific KNNClassifiers.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        y : ndarray
            Output (i.e. response) data matrix.

        transformer_id_to_transformers : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a transformer. This dictionary
        maps transformers to a particular transformer id.

        transformer_id_to_voters : dict
            A dictionary with keys of type obj corresponding to transformer ids
            and values of type obj corresponding to a voter class. This dictionary
            maps voter classes to a particular transformer id.

        Returns
        -------
        self : KNNClassifierDecider
            The object itself.
        """
        if not isinstance(self.classes, (list, np.ndarray)):
            if len(y) == 0:
                raise ValueError(
                    "Classification Decider classes undefined with no class labels fed to fit"
                )
            else:
                self.classes = np.unique(y)
        else:
            self.classes = np.array(self.classes)

        if self.k == None:
            self.k = int(np.log2(len(X)))
        X, y = check_X_y(X, y)

        # Because this instantiation relies on using the same transformers at train
        # and test time, we need to store them.
        self.transformer_ids = list(transformer_id_to_transformers.keys())
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters

        # Set up the samples, classes, transformer and tree counts.
        num_samples = len(X)
        num_classes = len(self.classes)
        num_transformers = len(self.transformer_ids)
        num_trees = len(self.transformer_id_to_transformers[0])

        # Get voter predictions of y
        yhats = self.ensemble_representations(X)

        self.knn = []
        temp_yhats = np.empty((num_samples, num_transformers))

        # Iterate across all trees.
        for bag_id in range(num_trees):
            tree_knns = []
            # Iterate across all classes.
            for class_idx in range(num_classes):
                # Add a new classifier for each class for each tree.
                tree_knns.append(KNeighborsClassifier(self.k, weights="distance", p=1))
                # Iterate across all transformers.
                for transformer_idx in range(num_transformers):
                    # Get the single class voter posterior of all transformers from each tree.
                    temp_yhats[:, transformer_idx] = yhats[
                        :, class_idx, transformer_idx, bag_id
                    ]

                # Create a class specific label and fit a KNNClassifier for a single class
                # with the voter posteriors from all transformers.
                class_label = np.array(
                    [1 if element == class_idx else 0 for element in y]
                )
                tree_knns[class_idx].fit(temp_yhats, class_label)

            # Append the fitted KNNClassifiers to the total list of KNNClassifiers.
            self.knn.append(tree_knns)

        self._is_fitted = True
        return self

    def predict_proba(self, X, transformer_ids=None):
        """
        Predicts the most likely class per input example.

        Uses each class-specific KNN to predict class labels for each bag using all
        voter posteriors from each transformer.
        Returns the class predictions for all examples. 

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        transformer_ids : list, default=None
            A list with all transformer ids. Defaults to None if no transformer ids
            are given.

        Returns
        -------
        y_hat : ndarray of shape [n_samples, n_classes]
            All predicted class labels per example.

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)

        # Set up the samples, classes, transformer and tree counts.
        num_samples = len(X)
        num_classes = len(self.classes)
        num_transformers = len(self.transformer_ids)
        num_trees = len(self.transformer_id_to_transformers[0])

        yhats = self.ensemble_representations(X)

        knn_out = np.empty((num_samples, num_classes, num_trees))
        temp_yhats = np.empty((num_samples, num_transformers))

        # Iterate across all trees.
        for bag_id in range(num_trees):
            # Iterate across all classes.
            for class_idx in range(num_classes):
                # Iterate across all transformers.
                for transformer_idx in range(num_transformers):
                    # Get all the voter posteriors from all transformers for each class and tree.
                    temp_yhats[:, transformer_idx] = yhats[
                        :, class_idx, transformer_idx, bag_id
                    ]
                # Use each knn for each class and tree to predict the probability from the average
                # voter posterior across all transformers.
                knn_out[:, class_idx, bag_id] = self.knn[bag_id][
                    class_idx
                ].predict_proba(temp_yhats)[:, 1]

        # Average the predicted posteriors across all trees and normalize them across all classes.
        mean_knn_out = np.mean(knn_out, axis=2)
        normalized = mean_knn_out / np.sum(mean_knn_out, axis=1, keepdims=True)
        return normalized

    def predict(self, X, transformer_ids=None):
        """
        Predicts the most likely class per input example using the predict_proba
        funtion. Returns the argmax across the classes.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        transformer_ids : list, default=None
            A list with all transformer ids. Defaults to None if no transformer ids
            are given.

        Returns
        -------
        y_hat : ndarray of shape [n_samples]
            Predicted class label per example

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)

        knn_out = self.predict_proba(X)
        return np.argmax(knn_out, axis=1)

    def is_fitted(self):
        """
        Method to check if decider is fitted.
        """

        return self._is_fitted

    def ensemble_representations(self, X):
        """
        Predicts posterior probabilities per input example with voters and ensemble them.

        Loops through each transformer and bag of transformers.
        Performs a transformation of the input data with the transformer.
        Gets a voter to map the transformed input data into a posterior distribution.
        Record the vote for each bag and each transformer.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        Returns
        -------
        yhats : ndarray of shape [n_samples, n_classes, num_transformers, n_bags]
            Posteriors per example for each transformer and bag.
        """
        # Set up the samples, classes, transformer and tree counts.
        num_samples = len(X)
        num_classes = len(self.classes)
        num_transformers = len(self.transformer_ids)
        num_trees = len(self.transformer_id_to_transformers[0])

        # Make output array.
        yhats = np.zeros((num_samples, num_classes, num_transformers, num_trees))

        # Iterate across all transformers.
        for transformer_id in (
            self.transformer_ids
            if self.transformer_ids is not None
            else self.transformer_id_to_voters.keys()
        ):
            # Iterate across all trees.
            for bag_id in range(num_trees):
                # Get a transformer from a specific tree.
                transformer = self.transformer_id_to_transformers[transformer_id][
                    bag_id
                ]

                # Transform data with transformer.
                X_transformed = transformer.transform(X)

                # Get specific voter from transformer and tree indices and predict
                # on the transformed data.
                voter = self.transformer_id_to_voters[transformer_id][bag_id]
                vote = voter.predict_proba(X_transformed).reshape(
                    num_samples, num_classes
                )

                # Add the vote to the output.
                yhats[:, :, transformer_id, bag_id] = vote

        # Return voter output of probabilities.
        return yhats


class KNNSimpleClassificationDecider(BaseClassificationDecider):
    """
    A class for a decider that uses a k-nearest neighbor (KNN) vote for classification.
    Simple implementation uses voter posteriors for all classes and all transformers
    for each bag.
    
    Parameters
    ----------
    k : int, default=None
        Number of neighbors for KNN voting.

    classes : list, default=[]
        List of final output classification labels of type obj.

    Attributes
    ----------
    _is_fitted : boolean, default=False
        Parameter to keep track of whether the model is fitted.

    transformer_ids : list
        A list of transformer ids.

    transformer_id_to_transformers_ : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a transformer. This dictionary
        maps transformers to a particular transformer id.

    transformer_id_to_voters_ : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a voter class. This dictionary
        maps voter classes to a particular transformer id.

    knn : list, default=[]
        A list of sklearn.KNNClassifiers with length equal to the number of bags.
    """

    def __init__(self, k=None, classes=[]):
        self.k = k
        self._is_fitted = False
        self.classes = classes

    def fit(self, X, y, transformer_id_to_transformers, transformer_id_to_voters):
        """
        Fits a KNNClassifier for each bag.

        Get the voter posteriors from ensemble_representations.
        Loop through each bag.
        Fit a KNNClassifier to the voter posteriors along all transformers and class
        for each bag.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        y : ndarray
            Output (i.e. response) data matrix.

        transformer_id_to_transformers : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a transformer. This dictionary
        maps transformers to a particular transformer id.

        transformer_id_to_voters : dict
            A dictionary with keys of type obj corresponding to transformer ids
            and values of type obj corresponding to a voter class. This dictionary
            maps voter classes to a particular transformer id.

        Returns
        -------
        self : KNNClassifierDecider
            The object itself.
        """
        if not isinstance(self.classes, (list, np.ndarray)):
            if len(y) == 0:
                raise ValueError(
                    "Classification Decider classes undefined with no class labels fed to fit"
                )
            else:
                self.classes = np.unique(y)
        else:
            self.classes = np.array(self.classes)

        if self.k == None:
            self.k = int(np.log2(len(X)))
        X, y = check_X_y(X, y)

        # Because this instantiation relies on using the same transformers at train
        # and test time, we need to store them.
        self.transformer_ids = list(transformer_id_to_transformers.keys())
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters

        # Set up the samples, classes, transformer and tree counts.
        num_samples = len(X)
        num_classes = len(self.classes)
        num_transformers = len(self.transformer_ids)
        num_trees = len(self.transformer_id_to_transformers[0])

        # Get voter predictions of y
        yhats = self.ensemble_representations(X)

        self.knn = []

        # Iterate across all representers.
        for bag_id in range(num_trees):
            # Make an empty vector.
            temp_yhats = np.empty((num_samples, num_classes * num_transformers))
            # Append a new KNN for each transformer with k neighbors.
            self.knn.append(KNeighborsClassifier(self.k, weights="distance", p=1))
            for transformer_idx in range(num_transformers):
                idx = num_classes * transformer_idx
                # Append all class probabilities from each transformer.
                temp_yhats[:, idx : (idx + num_classes)] = yhats[
                    :, :, transformer_idx, bag_id
                ]
            # Fit the representer specific knn with all class probabilities.
            self.knn[bag_id].fit(np.squeeze(temp_yhats), y)

        self._is_fitted = True
        return self

    def predict_proba(self, X, transformer_ids=None):
        """
        Predicts the most likely class per input example.

        Uses each KNN to predict class labels for each bag using all voter posteriors from each
        transformer. Returns the class predictions for all examples.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        transformer_ids : list, default=None
            A list with all transformer ids. Defaults to None if no transformer ids
            are given.

        Returns
        -------
        y_hat : ndarray of shape [n_samples, n_classes]
            All predicted class labels per example.

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)

        # Set up the samples, classes, transformer and tree counts.
        num_samples = len(X)
        num_classes = len(self.classes)
        num_transformers = len(self.transformer_ids)
        num_trees = len(self.transformer_id_to_transformers[0])

        yhats = self.ensemble_representations(X)

        knn_out = np.empty((num_samples, num_classes, num_trees))

        # Iterate across all trees.
        for bag_id in range(num_trees):
            temp_yhats = np.empty((num_samples, num_classes * num_transformers))
            for transformer_idx in range(num_transformers):
                idx = num_classes * transformer_idx
                temp_yhats[:, idx : (idx + num_classes)] = yhats[
                    :, :, transformer_idx, bag_id
                ]

            knn_out[:, :, bag_id] = self.knn[bag_id].predict_proba(
                np.squeeze(temp_yhats)
            )

        # Average the predicted posteriors across all trees and normalize them across all classes.
        mean_knn_out = np.mean(knn_out, axis=2)
        normalized = mean_knn_out / np.sum(mean_knn_out, axis=1, keepdims=True)

        return mean_knn_out

    def predict(self, X, transformer_ids=None):
        """
        Predicts the most likely class per input example using the predict_proba
        funtion. Returns the argmax across the classes.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        transformer_ids : list, default=None
            A list with all transformer ids. Defaults to None if no transformer ids
            are given.

        Returns
        -------
        y_hat : ndarray of shape [n_samples]
            Predicted class label per example

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)

        knn_out = self.predict_proba(X)
        return np.argmax(knn_out, axis=1)

    def is_fitted(self):
        """
        Method to check if decider is fitted.
        """

        return self._is_fitted

    def ensemble_representations(self, X):
        """
        Predicts posterior probabilities per input example with voters and ensemble them.

        Loops through each transformer and bag of transformers.
        Performs a transformation of the input data with the transformer.
        Gets a voter to map the transformed input data into a posterior distribution.
        Record the class votes for each bag and each transformer.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        Returns
        -------
        yhats : ndarray of shape [n_samples, n_classes, num_transformers, n_bags]
            Posteriors per example for each class for each transformer and bag.
        """
        # Set up the samples, classes, transformer and tree counts.
        num_samples = len(X)
        num_classes = len(self.classes)
        num_transformers = len(self.transformer_ids)
        num_trees = len(self.transformer_id_to_transformers[0])

        # Make output array.
        yhats = np.zeros((num_samples, num_classes, num_transformers, num_trees))

        # Iterate across all transformers.
        for transformer_id in (
            self.transformer_ids
            if self.transformer_ids is not None
            else self.transformer_id_to_voters.keys()
        ):
            # Iterate across all trees.
            for bag_id in range(num_trees):
                # Get a transformer from a specific tree.
                transformer = self.transformer_id_to_transformers[transformer_id][
                    bag_id
                ]

                # Transform data with transformer.
                X_transformed = transformer.transform(X)

                # Get specific voter from transformer and tree indices and predict
                # on the transformed data.
                voter = self.transformer_id_to_voters[transformer_id][bag_id]
                vote = voter.predict_proba(X_transformed).reshape(
                    num_samples, num_classes
                )

                # Add the vote to the output.
                yhats[:, :, transformer_id, bag_id] = vote

        # Return voter output of probabilities.
        return yhats


class KNNRegressionDecider(BaseDecider):
    """
    A class for a decider that uses a k-nearest neighbor (KNN) vote for regression.

    Parameters
    ----------
    k : list, default=[]
        List of neighbors for KNN voting.

    Attributes
    ----------
    _is_fitted : boolean, default=False
        Parameter to keep track of whether the model is fitted.

    transformer_ids : list
        A list of transformer ids.

    transformer_id_to_transformers_ : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a transformer. This dictionary
        maps transformers to a particular transformer id.

    transformer_id_to_voters_ : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a voter class. This dictionary
        maps voter classes to a particular transformer id.

    knn : list, default=[]
        A list of sklearn.KNNRegressors with length equal to the number of bags.
    """

    def __init__(self, k=None):
        self.k = k
        self._is_fitted = False

    def fit(self, X, y, transformer_id_to_transformers, transformer_id_to_voters):
        """
        Fits a KNNRegressor for each bag.

        Get the voter posteriors from ensemble_representations.
        Loop through each bag.
        Fit a KNNRegressor to the voter posteriors along all transformers for each bag.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        y : ndarray
            Output (i.e. response) data matrix.

        transformer_id_to_transformers_ : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a transformer. This dictionary
        maps transformers to a particular transformer id.

        transformer_id_to_voters_ : dict
            A dictionary with keys of type obj corresponding to transformer ids
            and values of type obj corresponding to a voter class. This dictionary
            maps voter classes to a particular transformer id.

        Returns
        -------
        self : KNNRegressionDecider
            The object itself.
        """
        X, y = check_X_y(X, y)
        n = len(y)
        if not self.k:
            self.k = min(16 * int(np.log2(n)), int(0.33 * n))

        # Because this instantiation relies on using the same transformers at train
        # and test time, we need to store them.
        self.transformer_ids = list(transformer_id_to_transformers.keys())
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters

        num_samples = len(X)
        num_trees = len(self.transformer_id_to_transformers[0])

        # Get the voter posteriors.
        yhats = self.ensemble_representations(X)
        self.knn = []

        # Fit a sklearn.KNeighborsRegressor for each bag.
        for bag_id in range(num_trees):
            self.knn.append(KNeighborsRegressor(self.k, weights="distance", p=1))
            self.knn[bag_id].fit(yhats[:, :, bag_id], y)

        self._is_fitted = True
        return self

    def predict(self, X, transformer_ids=None):
        """
        Predicts the regression output per input example.

        Uses each KNN to predict a vote each for bag.
        Returns the average prediction across all bags.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        transformer_ids : list, default=None
            A list with all transformer ids. Defaults to None if no transformer ids
            are given.

        Returns
        -------
        y_hat : ndarray of shape [n_samples]
            predicted class label per example

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)

        num_samples = len(X)
        num_trees = len(self.transformer_id_to_transformers[0])

        yhats = self.ensemble_representations(X)
        knn_out = np.empty((num_samples, num_trees))

        # Predict for each bag using the bag-specific KNNRegressor.
        for bag_id in range(num_trees):
            knn_out[:, bag_id] = self.knn[bag_id].predict(yhats[:, :, bag_id])

        # Return the mean prediction for each bag.
        return np.mean(knn_out, axis=1)

    def is_fitted(self):
        """
        Method to check if decider is fitted.
        """

        return self._is_fitted

    def ensemble_representations(self, X):
        """
        Predicts posterior probabilities per input example with voters and ensemble them.

        Loops through each transformer and bag of transformers.
        Performs a transformation of the input data with the transformer.
        Gets a voter to map the transformed input data into a posterior distribution.
        Record the vote for each bag and each transformer.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        Returns
        -------
        yhats : ndarray of shape [n_samples, num_transformers, n_bags]
            Posteriors per example for each transformer and bag.
        """
        n = len(X)
        num_transformers = len(self.transformer_ids)
        num_trees = len(self.transformer_id_to_transformers[0])

        yhats = np.zeros((n, num_transformers, num_trees))

        # Transformer IDs may not necessarily be {0, ..., num_transformers - 1}.
        for transformer_id in (
            self.transformer_ids
            if self.transformer_ids is not None
            else self.transformer_id_to_voters.keys()
        ):
            for bag_id in range(num_trees):
                # Get the specified transformer for this bag.
                transformer = self.transformer_id_to_transformers[transformer_id][
                    bag_id
                ]
                # Transform the data and get the voter.
                X_transformed = transformer.transform(X)
                voter = self.transformer_id_to_voters[transformer_id][bag_id]
                # Vote on the transformed data.
                yhats[:, transformer_id, bag_id] = voter.predict(X_transformed).reshape(
                    n
                )

        return yhats
