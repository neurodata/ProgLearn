import pytest
import numpy as np
from numpy import random as rng
from numpy.testing import assert_almost_equal, assert_warns, assert_raises
from sklearn.datasets import load_iris

from proglearn.oblique_tree import *


class TestObliqueSplitter:

    def test_sample_projmat(self):
        
        random_state = 0
        rng.seed(random_state)

        X = rng.rand(100, 100)
        y = np.zeros(100)

        density = 0.5
        proj_dims = [10, 20, 40, 60, 80]
        sample_inds = [np.linspace(0, 9, 10, dtype=int),
                       np.linspace(0, 19, 20, dtype=int), 
                       np.linspace(0, 39, 40, dtype=int), 
                       np.linspace(0, 59, 60, dtype=int), 
                       np.linspace(0, 79, 80, dtype=int)]

        n_sample_inds = [10, 20, 40, 60, 80]

        for pd in proj_dims:
            splitter = ObliqueSplitter(X, y, pd, density, random_state)

            for i in range(len(n_sample_inds)):
                si = sample_inds[i]
                n = n_sample_inds[i]

                proj_X, projmat = splitter.sample_proj_mat(si)
                assert n == proj_X.shape[0]
                assert pd == proj_X.shape[1]

    def test_score(self):

        random_state = 0
        rng.seed(random_state)

        X = rng.rand(11, 11)

        density = 0.5
        proj_dims = 5

        y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        splitter = ObliqueSplitter(X, y, proj_dims, density, random_state)

        score = splitter.score(y, 6)
        assert 0 == score

        score = splitter.score(y, 1)
        assert_almost_equal(5/11, score)

    def test_impurity(self):
       
        random_state = 0
        rng.seed(random_state)

        X = rng.rand(100, 100)

        density = 0.5
        proj_dims = 50

        y = np.zeros(100)
        for i in range(10):
            for j in range(10):
                y[10*i + j] = i
        
        splitter = ObliqueSplitter(X, y, proj_dims, density, random_state)
        
        # Impurity of one thing should be 0
        impurity = splitter.impurity([0])
        assert 0 == impurity
        
        # Impurity of one class should be 0
        impurity = splitter.impurity(np.linspace(0, 9, 10, dtype=int))
        assert 0 == impurity

        # Impurity of two different classes with equal number should be 0.5
        impurity = splitter.impurity(np.linspace(0, 19, 20, dtype=int))
        assert 0.5 == impurity

        # Impurity of all classes should be 10 * (1/10)(9/10) = 9/10
        impurity = splitter.impurity(np.linspace(0, 99, 100, dtype=int))
        assert_almost_equal(0.9, impurity)
    
    def test_split(self):

        random_state = 0
        rng.seed(random_state)

        X = rng.rand(100, 100)

        density = 0.5
        proj_dims = 50

        y = np.zeros(100)
        for i in range(10):
            for j in range(10):
                y[10*i + j] = i
        
        splitter = ObliqueSplitter(X, y, proj_dims, density, random_state)
        
        split_info = splitter.split(np.array([i for i in range(100)]))

class TestObliqueTree:

    def test_add_node(self):
        
        # Add a root node
        tree = ObliqueTree(None, 0, 0, 0, 0, 0)

        tree.add_node(0, False,
                      0, 0, False,
                      0, 0, None,
                      0, 0)

        # Add a regular node
        tree.add_node(0, False,
                      0, 0, False,
                      0, 0, None,
                      0, 0)

        # Add a leaf node
        tree.add_node(1, False,
                      0, 0, True,
                      0, 0, None,
                      0, 0)

        assert 3 == len(tree.nodes)
        assert 3 == tree.node_count


    def test_fit(self):

        data = load_iris() 
        clf = ObliqueTreeClassifier()
        clf.fit(data.data, data.target)

    def test_predict(self):
        Xtrain = np.random.rand(6, 5)
        ytrain = np.array([0, 0, 1, 1, 0, 1])
        tree = ObliqueTreeClassifier()
        tree.fit(Xtrain, ytrain)
        Xtest = np.random.rand(3, 5)
        preds = tree.predict(Xtest)

        assert len(preds) == len(Xtest)

        preds_proba = tree.predict_proba(Xtest)
        preds_log_proba = tree.predict_log_proba(Xtest)

        assert len(preds_proba) == len(Xtest)
        assert len(preds_log_proba) == len(Xtest)

        bool_inc = np.all(np.log(preds_proba) == preds_log_proba)

        assert bool_inc

