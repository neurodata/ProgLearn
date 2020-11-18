"""
Main Author: Will LeVine
Corresponding Email: levinewill@icloud.com
"""
import numpy as np

from .base import BaseClassificationDecider

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)


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
        self,
        X,
        y,
        transformer_id_to_transformers,
        transformer_id_to_voters,
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
