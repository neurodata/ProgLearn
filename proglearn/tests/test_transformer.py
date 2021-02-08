import pytest
import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_warns,
    assert_raises,
    assert_allclose,
)
from numpy import random as rng
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError

from proglearn.transformers import *


class TestTreeClassificationTransformer:
    def test_init(self):
        TreeClassificationTransformer()
        assert True

    def test_predict_without_fit(self):
        # Generate random data
        X = np.random.normal(0, 1, size=(100, 3))

        with pytest.raises(NotFittedError):
            trt = TreeClassificationTransformer()
            trt.transform(X)

    def test_correct_transformation(self):
        np.random.seed(1)

        trt = TreeClassificationTransformer()

        X = np.concatenate((np.zeros(100), np.ones(100))).reshape(-1, 1)
        y = np.concatenate((np.zeros(100), np.ones(100)))

        trt.fit(X, y)
        u1 = trt.transform(np.array([0]).reshape(1, -1))
        u2 = trt.transform(np.array([1]).reshape(1, -1))
        assert u1 != u2
