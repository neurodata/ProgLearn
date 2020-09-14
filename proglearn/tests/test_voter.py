#%%
import numpy as np

from sklearn.utils.validation import NotFittedError

from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleAverage
from proglearn.transformers import TreeClassificationTransformer, NeuralClassificationTransformer 
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter

import unittest

from numpy.testing import assert_allclose

def generate_data(n = 100):
    X = np.concatenate([np.zeros(n), np.ones(n)])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return X, y

class TestTreeClassificationVoter(unittest.TestCase):
    def test_initialize(self):
        TreeClassificationVoter()
        self.assertTrue(True)


    def test_predict_without_fit(self):
        X, y = generate_data()
        with self.assertRaises(NotFittedError):
            voter = TreeClassificationVoter()
            voter.vote(X)

    def test_correct_predict(self):
        X, y = generate_data()

        voter = TreeClassificationVoter()
        voter.fit(X, y)
        
        y_hat = np.argmax(voter.vote(X), axis = 1)

        assert_allclose(y, y_hat)
        
if __name__ == '__main__':
    unittest.main()