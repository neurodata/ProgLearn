'''
Main Author: Will LeVine
Corresponding Email: levinewill@icloud.com
'''
import numpy as np

from .base import BaseClassificationDecider

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)

from sklearn.utils.multiclass import type_of_target


class SimpleArgmaxAverage(BaseClassificationDecider):
    """
    Doc string here.
    """

    def __init__(self, classes=[]):
        self.classes = classes
        self._is_fitted = False

    def fit(
        self,
        X,
        y,
        transformer_id_to_transformers,
        transformer_id_to_voters,
        classes=None,
    ):
        if not isinstance(self.classes, (list, np.ndarray)):
            if len(y) == 0:
                raise ValueError("Classification Decider classes undefined with no class labels fed to fit")
            else:
                self.classes = np.unique(y)
        else:
            self.classes = np.array(self.classes)
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters
        
        self._is_fitted = True
        return self

    def predict_proba(self, X, transformer_ids=None):
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
                vote = voter.predict_proba(X_transformed)
                vote_per_bag_id.append(vote)
            vote_per_transformer_id.append(np.mean(vote_per_bag_id, axis=0))
        return np.mean(vote_per_transformer_id, axis=0)

    def predict(self, X, transformer_ids=None):
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
        Doc strings here.
        """

        return self._is_fitted
