"""
Main Author: Will LeVine
Corresponding Email: levinewill@icloud.com
"""
import numpy as np

from .base import BaseClassificationDecider

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)

from sklearn.utils.multiclass import type_of_target


class SimpleArgmaxAverage(BaseClassificationDecider):
    """
    A class for a decider that uses the average vote for classification. 
    Uses BaseClassificationDecider as a base class.

    Parameters:
    -----------
    classes : list, default=[]
        Defaults to an empty list of classes.

    Attributes (objects):
    -----------
    classes : list, default=[]
        Defaults to an empty list of classes.

    _is_fitted : boolean, default=False
        Boolean variable to see if the decider is fitted, defaults to False

    transformer_id_to_transformers : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a transformer. This dictionary 
        maps transformers to a particular transformer id.

    transformer_id_to_voters : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a voter class. This dictionary
        maps voter classes to a particular transformer id.

    Methods
    -----------
    fit(X, y, transformer_id_to_transformers, transformer_id_to_voters, classes=None)
        Fits tree classification to transformed data X with labels y.

    predict_proba(X, transformers_id=None)
        Predicts posterior probabilities given input data, X, for each class.

    predict(X, transformer, transformer_ids=None)
        Predicts the most likely class given input data X.

    is_fitted()
        Returns if the Decider has been fitted.
    """
    
    def __init__(self, classes=[]):
        self.classes = classes
        self._is_fitted = False

    def fit(
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

        classes : list, default=None
            A list of classes of type obj.

        Raises:
        -----------
        ValueError :
            When the labels have not been provided. len(y) == 0.

        Returns:
        ----------
        SimpleArgmaxAverage obj
            The ClassificationDecider object of class SimpleArgmaxAverage is returned.
        """
        self,
        X,
        y,
        transformer_id_to_transformers,
        transformer_id_to_voters,
        classes=None,
    ):
        if not isinstance(self.classes, (list, np.ndarray)):
            if len(y) == 0:
                raise ValueError(
                    "Classification Decider classes undefined with no class labels fed to fit"
                )
            else:
                self.classes = np.unique(y)
        else:
            self.classes = np.array(self.classes)
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters

        self._is_fitted = True
        return self

    def predict_proba(self, X, transformer_ids=None):
        """
        Predicts posterior probabilities per input example.

        Loops through each transformer and bag of transformers.
        Performs a transformation of the input data with the transformer.
        Gets a voter to map the transformed input data into a posterior distribution.
        Gets the mean vote per bag and append it to a vote per transformer id.
        Returns the average vote per transformer id.

        Parameters:
        -----------
        X : ndarray
            Input data matrix.

        transformer_ids : list, default=None
            A list with all transformer ids. Defaults to None if no transformer ids
            are given.
        
        Raises:
        -----------
        NotFittedError :
            When the model is not fitted.

        Returns:
        -----------
        Returns mean vote across transformer ids as an ndarray.
        """
        vote_per_transformer_id = []
        for transformer_id in (
            transformer_ids
            if transformer_ids is not None
            else self.transformer_id_to_voters.keys()
        ):
            if not self.is_fitted():
                msg = (
                    "This %(name)s instance is not fitted yet. Call 'fit' with "
                    "appropriate arguments before using this decider."
                )
                raise NotFittedError(msg % {"name": type(self).__name__})

            vote_per_bag_id = []
            for bag_id in range(
                len(self.transformer_id_to_transformers[transformer_id])
            ):
                transformer = self.transformer_id_to_transformers[transformer_id][
                    bag_id
                ]
                X_transformed = transformer.transform(X)
                voter = self.transformer_id_to_voters[transformer_id][bag_id]
                vote = voter.predict_proba(X_transformed)
                vote_per_bag_id.append(vote)
            vote_per_transformer_id.append(np.mean(vote_per_bag_id, axis=0))
        return np.mean(vote_per_transformer_id, axis=0)

    def predict(self, X, transformer_ids=None):
        """
        Predicts the most likely class per input example.

        Uses the predict_proba method to get the mean vote per id. 
        Returns the class with the highest vote.

        Parameters:
        -----------
        X : ndarray
            Input data matrix.

        transformer_ids : list, default=None
            A list with all transformer ids. Defaults to None if no transformer ids
            are given.

        Returns:
        -----------
        The class with the highest vote based on the argmax of the votes as an int.
        """
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this decider."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        vote_overall = self.predict_proba(X, transformer_ids=transformer_ids)
        return self.classes[np.argmax(vote_overall, axis=1)]

    def is_fitted(self):
        """
        Getter function to check if the decider is fitted.

        Returns:
        -----------
        Boolean class attribute _is_fitted.
        """
        return self._is_fitted