import unittest
import pytest
import numpy as np
import random

import keras

from proglearn.network import LifelongClassificationNetwork
from proglearn.transformers import NeuralClassificationTransformer
from proglearn.voters import KNNClassificationVoter
from proglearn.deciders import SimpleArgmaxAverage


class TestLifelongClassificationNetwork:
    def test_initialize(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())
        assert True

    def test_correct_default_transformer(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())
        assert l2n.pl_.default_transformer_class == NeuralClassificationTransformer

    def test_correct_default_voter(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())
        assert l2n.pl_.default_voter_class == KNNClassificationVoter

    def test_correct_default_decider(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())
        assert l2n.pl_.default_decider_class == SimpleArgmaxAverage

    def test_correct_default_kwargs_length(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())

        # transformer
        assert len(l2n.pl_.default_transformer_kwargs) == 5

        # voter
        assert len(l2n.pl_.default_voter_kwargs) == 0

        # decider
        assert len(l2n.pl_.default_decider_kwargs) == 0

    def test_correct_default_network_construction_proportion(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())
        assert l2n.default_network_construction_proportion == 0.67
