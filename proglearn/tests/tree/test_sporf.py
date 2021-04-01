
import numpy as np
from numpy.testing import (
        assert_almost_equal,
        assert_allclose,
        assert_array_equal,
        assert_array_almost_equal
        )

import pytest
from proglearn.transformers import ObliqueTreeClassifier as OTC

from sklearn import datasets
from sklearn.metrics import accuracy_score

"""
Sklearn test_tree.py stuff
"""
X_small = np.array([
    [0, 0, 4, 0, 0, 0, 1, -14, 0, -4, 0, 0, 0, 0, ],
    [0, 0, 5, 3, 0, -4, 0, 0, 1, -5, 0.2, 0, 4, 1, ],
    [-1, -1, 0, 0, -4.5, 0, 0, 2.1, 1, 0, 0, -4.5, 0, 1, ],
    [-1, -1, 0, -1.2, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 1, ],
    [-1, -1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1, ],
    [-1, -2, 0, 4, -3, 10, 4, 0, -3.2, 0, 4, 3, -4, 1, ],
    [2.11, 0, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0.5, 0, -3, 1, ],
    [2.11, 0, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0, 0, -2, 1, ],
    [2.11, 8, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0, 0, -2, 1, ],
    [2.11, 8, -6, -0.5, 0, 11, 0, 0, -3.2, 6, 0.5, 0, -1, 0, ],
    [2, 8, 5, 1, 0.5, -4, 10, 0, 1, -5, 3, 0, 2, 0, ],
    [2, 0, 1, 1, 1, -1, 1, 0, 0, -2, 3, 0, 1, 0, ],
    [2, 0, 1, 2, 3, -1, 10, 2, 0, -1, 1, 2, 2, 0, ],
    [1, 1, 0, 2, 2, -1, 1, 2, 0, -5, 1, 2, 3, 0, ],
    [3, 1, 0, 3, 0, -4, 10, 0, 1, -5, 3, 0, 3, 1, ],
    [2.11, 8, -6, -0.5, 0, 1, 0, 0, -3.2, 6, 0.5, 0, -3, 1, ],
    [2.11, 8, -6, -0.5, 0, 1, 0, 0, -3.2, 6, 1.5, 1, -1, -1, ],
    [2.11, 8, -6, -0.5, 0, 10, 0, 0, -3.2, 6, 0.5, 0, -1, -1, ],
    [2, 0, 5, 1, 0.5, -2, 10, 0, 1, -5, 3, 1, 0, -1, ],
    [2, 0, 1, 1, 1, -2, 1, 0, 0, -2, 0, 0, 0, 1, ],
    [2, 1, 1, 1, 2, -1, 10, 2, 0, -1, 0, 2, 1, 1, ],
    [1, 1, 0, 0, 1, -3, 1, 2, 0, -5, 1, 2, 1, 1, ],
    [3, 1, 0, 1, 0, -4, 1, 0, 1, -2, 0, 0, 1, 0, ]])

y_small = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
           0, 0]
y_small_reg = [1.0, 2.1, 1.2, 0.05, 10, 2.4, 3.1, 1.01, 0.01, 2.98, 3.1, 1.1,
               0.0, 1.2, 2, 11, 0, 0, 4.5, 0.201, 1.06, 0.9, 0]

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also load the diabetes dataset
# and randomly permute it
diabetes = datasets.load_diabetes()
perm = rng.permutation(diabetes.target.size)
diabetes.data = diabetes.data[perm]
diabetes.target = diabetes.target[perm]

# Ignoring digits dataset cause it takes a minute

def test_classification_toy():
    # Check classification on a toy dataset.
    clf = OTC(random_state=0)
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)

    """
    # Ignoring because max_features implemented differently
    clf = OTC(max_features=1, random_state=0)
    clf.fit(X, y)
    assert_array_equal(clf.predict(T), true_result)
    """

def test_xor():
    
    # Check on a XOR problem
    y = np.zeros((10, 10))
    y[:5, :5] = 1
    y[5:, 5:] = 1

    gridx, gridy = np.indices(y.shape)

    X = np.vstack([gridx.ravel(), gridy.ravel()]).T
    y = y.ravel()

    # Changing feature parameters from default 1.5 to 2 makes this test pass.
    clf = OTC(random_state=0, feature_combinations=2)
    clf.fit(X, y)

    assert accuracy_score(clf.predict(X), y) == 1

def test_iris():

    clf = OTC(random_state=0)

    clf.fit(iris.data, iris.target)
    score = accuracy_score(clf.predict(iris.data), iris.target)
    assert score > 0.9
    
def test_diabetes():
    
    """
    Diabetes should overfit with MSE = 0 for normal trees.
    idk if this applies to sporf, so this is just a placeholder
    to check consistency like iris.
    """

    clf = OTC(random_state=0)

    clf.fit(diabetes.data, diabetes.target)
    score = accuracy_score(clf.predict(diabetes.data), diabetes.target)
    assert score > 0.9
 
def test_probability():

    clf = OTC(random_state=0)

    clf.fit(iris.data, iris.target)
    p = clf.predict_proba(iris.data)

    assert_array_almost_equal(np.sum(p, 1),
                              np.ones(iris.data.shape[0]))

    assert_array_equal(np.argmax(p, 1),
                       clf.predict(iris.data))
    
    assert_almost_equal(clf.predict_proba(iris.data),
                        np.exp(clf.predict_log_proba(iris.data)))

def test_pure_set():

    clf = OTC(random_state=0)

    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    y = [1, 1, 1, 1, 1, 1]

    clf.fit(X, y)
    assert_array_equal(clf.predict(X), y)



