import pytest
import numpy as np
import random

from tensorflow import keras
from tensorflow.keras.layers import Dense
from sklearn.exceptions import NotFittedError

from proglearn.network import LifelongClassificationNetwork
from proglearn.transformers import NeuralClassificationTransformer
from proglearn.voters import KNNClassificationVoter
from proglearn.deciders import SimpleArgmaxAverage


def _generate_network():
    network = keras.Sequential()
    network.add(Dense(12, input_dim=8, activation="relu"))
    network.add(Dense(8, activation="relu"))
    network.add(Dense(2, activation="sigmoid"))

    return network


class TestLifelongClassificationNetwork:
    def test_initialize(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())
        assert True

    def test_correct_default_transformer(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())
        assert l2n.default_transformer_class == NeuralClassificationTransformer

    def test_correct_default_voter(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())
        assert l2n.default_voter_class == KNNClassificationVoter

    def test_correct_default_decider(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())
        assert l2n.default_decider_class == SimpleArgmaxAverage

    def test_correct_default_kwargs_length(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())

        # transformer
        assert len(l2n.default_transformer_kwargs) == 5

        # voter
        assert len(l2n.default_voter_kwargs) == 0

        # decider
        assert len(l2n.default_decider_kwargs) == 0

    def test_correct_default_network_construction_proportion(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())
        assert l2n.default_network_construction_proportion == 0.67

    def test_predict_without_fit(self):
        X = np.array([[0, 1, 0, 1, 0, 1, 0, 1]])
        X = np.tile(X, (100, 1))

        with pytest.raises(NotFittedError):
            l2n = LifelongClassificationNetwork(network=_generate_network(),)
            l2n.predict(X, task_id=0)

    def test_predict_proba_without_fit(self):
        X = np.array([[0, 1, 0, 1, 0, 1, 0, 1]])
        X = np.tile(X, (100, 1))

        with pytest.raises(NotFittedError):
            l2n = LifelongClassificationNetwork(network=_generate_network(),)
            l2n.predict_proba(X, task_id=0)

    def test_add_task(self):
        X = np.array([[0, 1, 0, 1, 0, 1, 0, 1]])
        X = np.tile(X, (100, 1))
        y = np.tile([0, 1], 50)

        l2n = LifelongClassificationNetwork(network=_generate_network(),)
        l2n.add_task(X, y)

    def test_add_transformer(self):
        X = np.array([[0, 1, 0, 1, 0, 1, 0, 1]])
        X = np.tile(X, (100, 1))
        y = np.tile([0, 1], 50)

        l2n = LifelongClassificationNetwork(network=_generate_network(),)
        l2n.add_transformer(X, y)
