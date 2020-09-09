import pytest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError

from proglearn.voters import KNNClassificationVoter
from proglearn.base import BaseVoter
        
def test_vote_without_fit():
    X = np.random.randn(100,3)
    
    with pytest.raises(NotFittedError):
        kcv = KNNClassificationVoter()
        kcv.vote(X)
    
def test_correct_vote():
    np.random.seed(0)


