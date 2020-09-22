import pytest
import unittest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError

from proglearn.transformers import TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter

class test_treeclassification(unittest.TestCase):
    
    def test_predict_without_fit(self):
        # Generate random data
        X = np.random.normal(0, 1, size=(100, 3))
    
        with pytest.raises(NotFittedError):
            trt = TreeClassificationTransformer()
            trt.transform(X)
    
        with pytest.raises(NotFittedError):
            trv = TreeClassificationVoter()
            trv.vote(X)
    
    
    def test_correct_transformation(self):
        np.random.seed(1)
    
        trt = TreeClassificationTransformer()
    
        X = np.concatenate((np.zeros(100), np.ones(100))).reshape(-1, 1)
        y = np.concatenate((np.zeros(100), np.ones(100)))
    
        trt.fit(X, y)
        u1 = trt.transform(np.array([0]).reshape(1, -1))
        u2 = trt.transform(np.array([1]).reshape(1, -1))
        assert u1 != u2
    
    
    def test_correct_vote(self):
        np.random.seed(3)
    
        X = np.concatenate((np.zeros(100), np.ones(100)))
        y = np.concatenate((np.zeros(100), np.ones(100)))
    
        trv = TreeClassificationVoter()
        trv.fit(X, y)
    
        X_test = np.ones(6)
        y_test = np.ones(6)
        assert_allclose(y_test, trv.vote(X_test)[:,-1], atol=1e-4)
    
if __name__ == '__main__':
    unittest.main()
