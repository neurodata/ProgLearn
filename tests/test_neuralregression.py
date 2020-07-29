import pytest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError

from proglearn.transformers import NeuralRegressionTransformer
from proglearn.voters import NeuralRegressionVoter

from tensorflow import keras
from tensorflow.random import set_seed


def initialize_transformer():
    network = keras.Sequential()
    network.add(keras.layers.Dense(2, activation="relu", input_shape=(3,)))
    network.add(keras.layers.Dense(1, activation="relu"))

    return NeuralRegressionTransformer(network=network)


def test_predict_without_fit():
    # Generate random data
    X = np.random.normal(0, 1, size=(100, 3))

    with pytest.raises(NotFittedError):
        nrt = initialize_transformer()
        nrt.transform(X)

    with pytest.raises(NotFittedError):
        nrv = NeuralRegressionVoter()
        nrv.vote(X)


def test_correct_transformation():
    np.random.seed(1)
    set_seed(2)  # tensorflow

    nrt = initialize_transformer()
    nrt._is_fitted = True

    x = np.ones((1, 3))
    u1 = np.array([[0, 0.69286364]])

    u2 = nrt.transform(x)

    assert_allclose(u1, u2)


def test_correct_vote():
    np.random.seed(3)
    set_seed(4)  # tensorflow

    X = np.random.randn(1000, 3)
    Y = np.sum(X, axis=1).reshape((1000, 1))

    nrv = NeuralRegressionVoter(epochs=30, lr=1)
    nrv.fit(X, Y)

    X_test = np.ones((5, 3))
    Y_test = 3 * np.ones((5, 1))

    assert_allclose(Y_test, nrv.vote(X_test), atol=1e-4)
