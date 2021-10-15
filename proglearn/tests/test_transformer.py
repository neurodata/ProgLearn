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
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from proglearn.transformers import *


def _generate_network():
    network = keras.Sequential()
    network.add(
        layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=np.shape(x_all)[1:],
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=254,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )

    network.add(layers.Flatten())
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(2000, activation="relu"))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(2000, activation="relu"))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(units=10, activation="softmax"))

    return network


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


class TestNeuralClassificationTransformer:
    def test_init(self):
        NeuralClassificationTransformer(
            network=_generate_network(), euclidean_layer_idx=-2, optimizer=Adam(3e-4)
        )
        assert True

    def test_predict_without_fit(self):
        # Generate random data
        X = np.random.normal(0, 1, size=(100, 3))

        with pytest.raises(NotFittedError):
            trt = NeuralClassificationTransformer(
                network=_generate_network(),
                euclidean_layer_idx=-2,
                optimizer=Adam(3e-4),
            )
            trt.transform(X)

    def test_correct_transformation(self):
        np.random.seed(1)

        trt = NeuralClassificationTransformer(
            network=_generate_network(), euclidean_layer_idx=-2, optimizer=Adam(3e-4)
        )

        X = np.concatenate((np.zeros(100), np.ones(100))).reshape(-1, 1)
        y = np.concatenate((np.zeros(100), np.ones(100)))

        trt.fit(X, y)
        u1 = trt.transform(np.array([0]).reshape(1, -1))
        u2 = trt.transform(np.array([1]).reshape(1, -1))
        assert u1 != u2
