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
        assert l2n.pl.default_transformer_class == NeuralClassificationTransformer
        
    def test_correct_default_voter(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())
        assert l2n.pl.default_voter_class == KNNClassificationVoter
        
    def test_correct_default_decider(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())
        assert l2n.pl.default_decider_class == SimpleArgmaxAverage
        
    def test_correct_default_kwargs(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())
        
        #transformer 
        assert l2f.pl.default_transformer_kwargs == {
            "network": self.network,
            "euclidean_layer_idx": -2,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "fit_kwargs": {
                "epochs": self.epochs,
                "callbacks": [EarlyStopping(patience=5, monitor="val_loss")],
                "verbose": self.verbose,
                "validation_split": 0.33,
                "batch_size": self.batch_size
            },
        }
        
        #voter
        assert l2f.pl.default_voter_kwargs == {}
        
        #decider
        assert l2f.pl.default_decider_kwargs == {}
        
    def test_correct_default_transformer_voter_decider_split(self):
        l2n = LifelongClassificationNetwork(keras.Sequential())
        assert l2n.default_transformer_voter_decider_split == [0.67, 0.33, 0]
