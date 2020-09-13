import unittest
import pytest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError

from proglearn.voters import KNNClassificationVoter
from proglearn.base import BaseVoter
    
class TestKNNClassificationVoter(unittest.TestCase):

    
    def test_vote_without_fit(self):
        X = np.random.randn(100,3)
    
        with self.assertRaises(NotFittedError):
            kcv = KNNClassificationVoter(3)
            kcv.vote(X)


    def test_correct_vote(self):
        np.random.seed(0)
        #set_seed(4)  # tensorflow

        X = np.concatenate((np.zeros(100), np.ones(100))).reshape(-1,1)
        Y = np.concatenate((np.zeros(100), np.ones(100)))

        kcv = KNNClassificationVoter(3)
        kcv.fit(X, Y)

        X_test = np.ones(6).reshape(-1,1)
        Y_test = np.concatenate((np.zeros((6,1)), np.ones((6,1))), axis = 1)

        assert_allclose(Y_test, kcv.vote(X_test), atol=1e-4)
        

