import pytest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError

from proglearn.deciders import KNNRegressionDecider, LinearRegressionDecider
from proglearn.base import BaseTransformer, BaseVoter


def test_predict_without_fit():
    # Generate random data
    X = np.random.normal(0, 1, size=(100, 3))

    with pytest.raises(NotFittedError):
        krd = KNNRegressionDecider()
        krd.predict(X)

    with pytest.raises(NotFittedError):
        lrd = LinearRegressionDecider()
        lrd.predict(X)


class IdentityTransformer(BaseTransformer):
    def __init__(self):
        self._is_fitted = False

    def fit(self):
        self._is_fitted = True

    def transform(self, X):
        return X

    def is_fitted(self):
        return self._is_fitted


class IdentityVoter(BaseVoter):
    def __init__(self, index):
        self._is_fitted = False
        self.index = index

    def fit(self):
        self._is_fitted = True

    def vote(self, X):
        n = len(X)
        return X[:, self.index].reshape(n)

    def is_fitted(self):
        return self._is_fitted


def test_correct_decision():
    np.random.seed(3)

    X = 0.1 * np.random.randn(2000, 3) + 1
    Y = np.sum(X, axis=1).reshape((2000, 1))

    lrd = LinearRegressionDecider()
    krd = KNNRegressionDecider()

    transformer_id_to_transformers = {
        "0": [IdentityTransformer()],
        "1": [IdentityTransformer()],
        "2": [IdentityTransformer()],
    }
    transformer_id_to_voters = {
        "0": [IdentityVoter(0)],
        "1": [IdentityVoter(1)],
        "2": [IdentityVoter(2)],
    }

    kwargs = {
        "transformer_id_to_transformers": transformer_id_to_transformers,
        "transformer_id_to_voters": transformer_id_to_voters,
        "X": X,
    }
    lrd.fit(Y, **kwargs)
    krd.fit(Y, **kwargs)

    X_test = np.ones((5, 3))
    Y_test = 3 * np.ones(5)

    assert_allclose(Y_test, lrd.predict(X_test), atol=1e-4)
    assert_allclose(Y_test, krd.predict(X_test), atol=1e-2)
