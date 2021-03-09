import unittest
import pytest
import numpy as np
import random

from proglearn.forest import LifelongClassificationForest, UncertaintyForest
from proglearn.transformers import TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter
from proglearn.deciders import SimpleArgmaxAverage


class TestLifelongClassificationForest:
    def test_initialize(self):
        l2f = LifelongClassificationForest()
        assert True

    def test_correct_default_transformer(self):
        l2f = LifelongClassificationForest()
        assert l2f.pl_.default_transformer_class == TreeClassificationTransformer

    def test_correct_default_voter(self):
        l2f = LifelongClassificationForest()
        assert l2f.pl_.default_voter_class == TreeClassificationVoter

    def test_correct_default_decider(self):
        l2f = LifelongClassificationForest()
        assert l2f.pl_.default_decider_class == SimpleArgmaxAverage

    def test_correct_default_kwargs(self):
        l2f = LifelongClassificationForest()

        # transformer
        assert l2f.pl_.default_transformer_kwargs == {}

        # voter
        assert len(l2f.pl_.default_voter_kwargs) == 1
        assert "kappa" in list(l2f.pl_.default_voter_kwargs.keys())
        assert l2f.pl_.default_voter_kwargs["kappa"] == np.inf

        # decider
        assert l2f.pl_.default_decider_kwargs == {}

    def test_correct_default_n_estimators(self):
        l2f = LifelongClassificationForest()
        assert l2f.default_n_estimators == 100

    def test_correct_true_initilization_finite_sample_correction(self):
        l2f = LifelongClassificationForest(default_kappa=np.inf)
        assert l2f.pl_.default_voter_kwargs == {"kappa": np.inf}


# Test Uncertainty Forest


def test_uf_accuracy():
    uf = UncertaintyForest()
    X = np.ones((20, 4))
    X[10:] *= -1
    y = [0] * 10 + [1] * 10
    uf = uf.fit(X, y)
    np.testing.assert_array_equal(uf.predict(X), y)


@pytest.mark.parametrize("max_depth", [1, None])
@pytest.mark.parametrize("max_features", [2, 0.5, "auto", "sqrt", "log2"])
@pytest.mark.parametrize("poisson_sampler", [False, True])
def test_decision_tree_params(max_depth, max_features, poisson_sampler):
    uf = UncertaintyForest(
        max_depth=max_depth, max_features=max_features, poisson_sampler=poisson_sampler
    )
    X = np.ones((12, 20))
    X[6:] *= -1
    y = [0] * 6 + [1] * 6
    uf = uf.fit(X, y)

    assert uf.n_estimators == len(uf.estimators_)
    depths = [est.max_depth for est in uf.estimators_]
    assert all(np.asarray(depths) == max_depth)

    features = [est.max_features for est in uf.estimators_]
    if poisson_sampler:
        assert not all(np.asarray(features) == features[0])
    else:
        assert all(np.asarray(features) == max_features)


def test_parallel_trees():
    uf = UncertaintyForest(n_jobs=2)
    X = np.random.normal(0, 1, (100, 2))
    X[:50] *= -1
    y = [0] * 50 + [1] * 50
    uf = uf.fit(X, y)


def test_max_samples():
    max_samples_list = [8, 0.5, None]
    depths = []
    X = np.random.normal(0, 1, (100, 2))
    X[:50] *= -1
    y = [0, 1] * 50
    for ms in max_samples_list:
        uf = UncertaintyForest(n_estimators=1, max_samples=ms)
        uf = uf.fit(X, y)
        depths.append(uf.estimators_[0].get_depth())

    assert all(np.diff(depths) > 0)

# @pytest.mark.parametrize("signal_ranks", [None, 2])
def test_uf_params():
    pass
