# from the template test in sklearn:
# https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/tests/test_template.py

import math
import re

import numpy as np
import pytest
from sklearn import datasets, metrics
from sklearn.utils.validation import check_random_state

from proglearn.morf import Conv2DSplitter

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [0, 0, 0, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [0, 1, 1]

# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# also load the boston dataset
# and randomly permute it
boston = datasets.load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]


def test_convolutional_splitter():
    n = 50
    height = 40
    d = 40
    X = np.ones((n, height, d))
    y = np.ones((n,))
    y[:25] = 0

    splitter = Conv2DSplitter(
        X,
        y,
        max_features=1,
        feature_combinations=1.5,
        random_state=random_state,
        image_height=height,
        image_width=d,
        patch_height_max=2,
        patch_height_min=2,
    )
