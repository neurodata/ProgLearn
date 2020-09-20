#%%
import numpy as np

from sklearn.utils.validation import NotFittedError

from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import TreeClassificationTransformer, NeuralClassificationTransformer
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter

import pytest

from numpy import testing

def generate_data(n = 100):
    X = np.concatenate([np.zeros(n), np.ones(n)])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return X, y

class TestTreeClassificationVoter:
    def test_initialize(self):
        TreeClassificationVoter()
        assert


    def test_predict_without_fit(self):
        X, y = generate_data()
        assert_raises(NotFittedError, TreeClassificationVoter().vote, X)

    def test_correct_predict(self):
        X, y = generate_data()

        voter = TreeClassificationVoter()
        voter.fit(X, y)

        y_hat = np.argmax(voter.vote(X), axis = 1)

        testing.assert_allclose(y, y_hat)
