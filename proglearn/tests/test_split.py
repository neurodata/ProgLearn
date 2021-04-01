import numpy as np
from numpy.testing import (
        assert_almost_equal,
        assert_allclose
)
import pytest
from split import BaseObliqueSplitter as BOS

class TestBaseSplitter:

    def test_argsort(self):
        b = BOS()

        # Ascending array
        y = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        idx = b.test_argsort(y)
        assert_allclose(y, idx)

        # Descending array
        y = np.array([4, 3, 2, 1, 0], dtype=np.float64)
        idx = b.test_argsort(y)
        assert_allclose(y, idx)

        # Array with repeated values
        y = np.array([1, 1, 1, 0, 0], dtype=np.float64)
        idx = b.test_argsort(y)
        assert_allclose([3, 4, 0, 1, 2], idx)

    def test_argmin(self):

        b = BOS()

        X = np.ones((5, 5), dtype=np.float64)
        X[3, 4] = 0
        (i, j) = b.test_argmin(X)
        assert 3 == i
        assert 4 == j

    def test_impurity(self):
        
        """
        First 2 
        Taken from SPORF's fpGiniSplitTest.h
        """

        b = BOS()
        
        y = np.ones(6, dtype=np.float64) * 4
        imp = b.test_impurity(y)
        assert 0 == imp

        y[:3] = 2
        imp = b.test_impurity(y)
        assert 0.5 == imp

        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.float64)
        imp = b.test_impurity(y)
        assert_almost_equal((2/3), imp)

    def test_score(self):

        b = BOS()

        y = np.ones(10, dtype=np.float64)
        y[:5] = 0
        s = b.test_score(y, 5)
        assert 0 == s

        y = np.ones(9, dtype=np.float64)
        y[:3] = 0
        y[6:] = 2
        s = b.test_score(y, 3)
        assert_almost_equal((1/3), s)

    def test_halfSplit(self):
        
        b = BOS()

        y = np.ones(100, dtype=np.float64) * 4
        y[:50] = 2

        X = np.ones((100, 1), dtype=np.float64) * 10
        X[:50] = 5

        idx = np.array([i for i in range(100)], dtype=np.intc)

        (feat, 
         thresh, 
         left_imp, 
         left_idx, 
         right_imp, 
         right_idx,
         improvement) = b.best_split(X, y, idx)
        assert thresh == 7.5
        assert left_imp == 0
        assert right_imp == 0
        assert feat == 0

    def test_oneOffEachEnd(self):

        b = BOS()

        y = np.ones(6, dtype=np.float64) * 4
        y[0] = 1

        X = np.ones((6, 1), dtype=np.float64) * 10
        X[0] = 5

        idx = np.array([i for i in range(6)], dtype=np.intc)
        
        (feat1, 
         thresh1, 
         left_imp1, 
         left_idx, 
         right_imp1, 
         right_idx,
         improvement) = b.best_split(X, y, idx)

        y = np.ones(6, dtype=np.float64) * 4
        y[-1] = 1

        X = np.ones((6, 1), dtype=np.float64) * 10
        X[-1] = 5
        
        (feat2, 
         thresh2, 
         left_imp2, 
         left_idx, 
         right_imp2, 
         right_idx,
         improvement) = b.best_split(X, y, idx)
        assert feat1 == feat2
        assert thresh1 == thresh2
        assert left_imp1 + right_imp1 == left_imp2 + right_imp2

    def test_secondFeature(self):

        pass


    



