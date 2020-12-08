import numpy as np

from .base import BaseDecider, ClassificationDecider

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)

from sklearn.utils.multiclass import type_of_target


class SimpleAverage(ClassificationDecider):
    """
    Doc string here.
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
        self.classes = self.classes if len(self.classes) > 0 else np.unique(y)
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters

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
                vote = voter.vote(X_transformed)
                vote_per_bag_id.append(vote)
            vote_per_transformer_id.append(np.mean(vote_per_bag_id, axis=0))
        return np.mean(vote_per_transformer_id, axis=0)

    def predict(self, X, transformer_ids=None):
        vote_overall = self.predict_proba(X, transformer_ids=transformer_ids)
        return self.classes[np.argmax(vote_overall, axis=1)]


class KNNRegressionDecider(BaseDecider):
    """
    Doc string here.
    """

    def __init__(self, k=None):
        self.k = k
        self._is_fitted = False

    def fit(self, X, y, transformer_id_to_transformers, transformer_id_to_voters):
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

        yhats = self.ensemble_represetations(X)
        self.knn = []

        for bag_id in range(num_trees):
            self.knn.append(KNeighborsRegressor(self.k, weights="distance", p=1))
            self.knn[bag_id].fit(yhats[:,:,bag_id], y)

        self._is_fitted = True
        return self

    def predict(self, X, transformer_ids=None):
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)

        num_samples = len(X)
        num_trees = len(self.transformer_id_to_transformers[0])

        yhats = self.ensemble_represetations(X)
        knn_out = np.empty((num_samples, num_trees))
        
        for bag_id in range(num_trees):
            knn_out[:,bag_id] = self.knn[bag_id].predict(yhats[:,:,bag_id])
        
        return np.mean(knn_out, axis=1)

    def is_fitted(self):
        """
        Doc strings here.
        """

        return self._is_fitted

    def ensemble_represetations(self, X):
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
                transformer = self.transformer_id_to_transformers[transformer_id][bag_id]
                X_transformed = transformer.transform(X)
                voter = self.transformer_id_to_voters[transformer_id][bag_id]
                yhats[:, transformer_id, bag_id] = voter.vote(X_transformed).reshape(n)

        return yhats


class LinearRegressionDecider(BaseDecider):
    """
    Doc string here.
    """

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_
        self._is_fitted = False

    def fit(
        self, X, y, transformer_id_to_transformers, transformer_id_to_voters,
    ):
        X, y = check_X_y(X, y)

        # Because this instantiation relies on using the same transformers at train
        # and test time, we need to store them.
        self.transformer_ids = list(transformer_id_to_transformers.keys())
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters

        yhats = self.ensemble_represetations(X)

        self.ridge = Ridge(self.lambda_)
        self.ridge.fit(yhats, y)

        self._is_fitted = True
        return self

    def predict(self, X, transformer_ids=None):
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)

        yhats = self.ensemble_represetations(X)

        return self.ridge.predict(yhats)

    def is_fitted(self):
        """
        Doc strings here.
        """

        return self._is_fitted

    def ensemble_represetations(self, X):
        n = len(X)

        # transformer_ids input is ignored - you can only use the transformers
        # you are trained on.
        yhats = np.zeros((n, len(self.transformer_ids)))

        # Transformer IDs may not necessarily be {0, ..., num_transformers - 1}.
        for i in range(len(self.transformer_ids)):

            transformer_id = self.transformer_ids[i]

            # The zero index is for the 'bag_id' as in random forest,
            # where multiple transformers are bagged together in each hypothesis.
            transformer = self.transformer_id_to_transformers[transformer_id][0]
            X_transformed = transformer.transform(X)
            voter = self.transformer_id_to_voters[transformer_id][0]
            yhats[:, i] = voter.vote(X_transformed).reshape(n)

        return yhats