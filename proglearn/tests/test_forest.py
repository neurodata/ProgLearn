from proglearn.deciders import SimpleArgmaxAverage
from proglearn.voters import TreeClassificationVoter
from proglearn.transformers import TreeClassificationTransformer
from proglearn.forest import LifelongClassificationForest, UncertaintyForest
import random
import numpy as np
import time
import pytest
import unittest
import pprint
import sys
pprint.pprint(sys.path)


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
    uf = UncertaintyForest(n_estimators=100, n_jobs=1,
                           max_features=1.0, tree_construction_proportion=0.5)
    uf_parallel = UncertaintyForest(
        n_estimators=100, n_jobs=10, max_features=1.0, tree_construction_proportion=0.5)
    X = np.random.normal(0, 1, (1000, 100))
    y = [0, 1] * (len(X) // 2)

    time_start = time.time()
    uf.fit(X, y)
    time_diff = time.time() - time_start

    time_start = time.time()
    uf_parallel.fit(X, y)
    time_parallel_diff = time.time() - time_start

    assert time_parallel_diff / time_diff < 0.9


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

@pytest.mark.parametrize("honest_prior", ["empirical", "uniform", "ignore"])
def test_honest_prior(honest_prior):
    X = np.random.normal(0, 1, (60, 2))
    X[:30] *= -1
    y = [0, 0, 1] * 20
    uf = UncertaintyForest(n_estimators=5, honest_prior=honest_prior)
    uf = uf.fit(X, y)
    if honest_prior == 'uniform':
        assert all([len(set(voter.prior_posterior_)) == 1 for voter in uf.voters_])
    elif honest_prior in ('ignore', 'empirical'):
        assert all([np.diff(voter.prior_posterior_) < 0 for voter in uf.voters_])

# @pytest.mark.parametrize("signal_ranks", [None, 2])
def test_uf_params():
    pass
