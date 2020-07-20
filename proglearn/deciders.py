import numpy as np

from .base import BaseDecider

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge

from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)

import keras


class SimpleAverage(BaseDecider):
    """
    Doc string here.
    """

    def __init__(self, classes=[]):
        self.classes = classes

    def fit(
        self,
        y,
        transformer_id_to_transformers,
        classes=None,
        transformer_id_to_voters=None,
        X=None,
    ):
        self.classes = self.classes if len(self.classes) > 0 else np.unique(y)
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters
        return self

    def predict(self, X, transformer_ids=None):
        vote_per_transformer_id = []
        for transformer_id in (
            transformer_ids if transformer_ids else self.transformer_id_to_voters.keys()
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
        vote_overall = np.mean(vote_per_transformer_id, axis=0)
        return self.classes[np.argmax(vote_overall, axis=1)]


class KNNRegressionDecider(BaseDecider):
    """
    Doc string here.
    """

    def __init__(self, k=None):
        self.k = k
        self._is_fitted = False

    def fit(
        self,
        y,
        transformer_id_to_transformers,
        classes=None,
        transformer_id_to_voters=None,
        X=None,
    ):
        X, y = check_X_y(X, y)
        n = len(y)
        if not self.k:
            self.k = min(16 * int(np.log2(n)), int(0.33 * n))

        # Because this instantiation relies on using the same transformers at train
        # and test time, we need to store them.
        self.transformer_ids = list(transformer_id_to_transformers.keys())
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters

        yhats = self.ensemble_represetations(X)

        self.knn = KNeighborsRegressor(self.k, weights="distance", p=1)
        self.knn.fit(yhats, y)

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

        return self.knn.predict(yhats)

    def is_fitted(self):
        """
        Doc strings here.
        """

        return self._is_fitted

    def ensemble_represetations(self, X):
        n = len(X)

        # transformer_ids input is ignored - you can only use
        # the transformers you are trained on.
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


class LinearRegressionDecider(BaseDecider):
    """
    Doc string here.
    """

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_
        self._is_fitted = False

    def fit(
        self,
        y,
        transformer_id_to_transformers,
        classes=None,
        transformer_id_to_voters=None,
        X=None,
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


class NeuralRegressionDecider(BaseDecider):
    """
    Doc string here.
    """

    def __init__(
        self,
        build_network,
        representation_size,
        optimizer,
        loss="mae",
        compile_kwargs={"metrics": ["MAPE", "MAE"]},
        fit_kwargs={
            "epochs": 100,
            "callbacks": [keras.callbacks.EarlyStopping(patience=5, monitor="val_MAE")],
            "verbose": False,
            "validation_split": 0.33,
        },
    ):
        self.build_network = build_network
        self.representation_size = representation_size
        self.optimizer = optimizer
        self.loss = loss
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs
        self._is_fitted = False
        self.num_tasks = 0

    def fit(
        self,
        y,
        transformer_id_to_transformers,
        classes=None,
        transformer_id_to_voters=None,
        X=None,
    ):

        X, y = check_X_y(X, y)

        # Because this instantiation relies on using the same transformers at train
        # and test time, we need to store them.
        self.transformer_ids = list(transformer_id_to_transformers.keys())
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters

        X_concat = self.ensemble_represetations(X)

        self.network = self.build_network(
            len(self.transformer_ids) * self.representation_size
        )

        # Build network that maps them to y.
        self.network.compile(
            loss=self.loss, optimizer=self.optimizer, **self.compile_kwargs
        )
        self.network.fit(X_concat, y, **self.fit_kwargs)

        self._is_fitted = True
        self.num_tasks += 1
        return self

    def predict(self, X, transformer_ids=None):
        if not self.is_fitted():
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)
        X_concat = self.ensemble_represetations(X)

        return self.network.predict(X_concat)

    def is_fitted(self):
        """
        Doc strings here.
        """

        return self._is_fitted

    def ensemble_represetations(self, X):
        # Form X_concat, n-by-num_transformers*num_representation units matrix
        # that act as the "new X". Here, n is the training set size of the
        # particular task that this is deciding on.
        X_concat = []

        for transformer_id in self.transformer_id_to_voters:
            # The zero index is for the 'bag_id' as in random forest,
            # where multiple transformers are bagged together in each hypothesis.
            transformer = self.transformer_id_to_transformers[transformer_id][0]
            X_transformed = transformer.transform(X)
            X_concat.append(X_transformed)

        return np.concatenate(X_concat, axis=1)

