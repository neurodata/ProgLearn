import pytest
import numpy as np
import random
from sklearn.exceptions import NotFittedError

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
        assert l2f.default_transformer_class == TreeClassificationTransformer

    def test_correct_default_voter(self):
        l2f = LifelongClassificationForest()
        assert l2f.default_voter_class == TreeClassificationVoter

    def test_correct_default_decider(self):
        l2f = LifelongClassificationForest()
        assert l2f.default_decider_class == SimpleArgmaxAverage

    def test_correct_default_kwargs(self):
        l2f = LifelongClassificationForest()

        # transformer
        assert l2f.default_transformer_kwargs == {}

        # voter
        assert len(l2f.default_voter_kwargs) == 1
        assert "kappa" in list(l2f.default_voter_kwargs.keys())
        assert l2f.default_voter_kwargs["kappa"] == np.inf

        # decider
        assert l2f.default_decider_kwargs == {}

    def test_correct_default_n_estimators(self):
        l2f = LifelongClassificationForest()
        assert l2f.default_n_estimators == 100

    def test_correct_true_initilization_finite_sample_correction(self):
        l2f = LifelongClassificationForest(default_kappa=np.inf)
        assert l2f.default_voter_kwargs == {"kappa": np.inf}

    def test_add_transformer(self):
        X = np.concatenate((np.zeros(100), np.ones(100))).reshape(-1, 1)
        y = np.concatenate((np.zeros(100), np.ones(100)))

        l2f = LifelongClassificationForest()
        l2f.add_transformer(X, y)

        assert np.array_equiv(l2f.transformer_id_to_X[0], X)
        assert np.array_equiv(l2f.transformer_id_to_y[0], y)

    def test_predict_without_fit(self):
        X = np.random.normal(0, 1, size=(100, 3))

        with pytest.raises(NotFittedError):
            l2f = LifelongClassificationForest()
            l2f.predict(X, task_id=0)

    def test_predict(self):
        np.random.seed(1)

        l2f = LifelongClassificationForest()

        X = np.concatenate((np.zeros(100), np.ones(100))).reshape(-1, 1)
        y = np.concatenate((np.zeros(100), np.ones(100)))

        l2f.add_task(X, y)
        u1 = l2f.predict(np.array([0]).reshape(1, -1), task_id=0)
        u2 = l2f.predict(np.array([1]).reshape(1, -1), task_id=0)
        assert u1 != u2

        u1 = l2f.predict(np.array([0]).reshape(1, -1), task_id=0)
        u2 = l2f.predict(np.array([0]).reshape(1, -1), task_id=0)
        assert u1 == u2

    def test_predict_proba(self):
        np.random.seed(1)

        l2f = LifelongClassificationForest()

        X = np.concatenate((np.zeros(100), np.ones(100))).reshape(-1, 1)
        y = np.concatenate((np.zeros(100), np.ones(100)))

        l2f.add_task(X, y)
        u1 = l2f.predict_proba(np.array([0]).reshape(1, -1), task_id=0)
        u2 = l2f.predict_proba(np.array([1]).reshape(1, -1), task_id=0)
        assert not np.array_equiv(u1, u2)

        u1 = l2f.predict_proba(np.array([0]).reshape(1, -1), task_id=0)
        u2 = l2f.predict_proba(np.array([0]).reshape(1, -1), task_id=0)
        assert np.array_equiv(u1, u2)

    #def test_update_task(self):
    #    np.random.seed(1)

    #    l2f = LifelongClassificationForest()

    #    X = np.concatenate((np.zeros(100), np.ones(100))).reshape(-1, 1)
    #    y = np.concatenate((np.zeros(100), np.ones(100)))

    #    l2f.add_task(X, y)
    #    u1 = l2f.predict_proba(np.array([0]).reshape(1, -1), task_id=0)
    #    u2 = l2f.predict_proba(np.array([1]).reshape(1, -1), task_id=0)

    #    X2 = np.concatenate((np.zeros(100), np.ones(100))).reshape(-1, 1)
    #    y2 = np.concatenate((np.zeros(100), np.ones(100)))

    #    X3 = np.concatenate((X, X2))
    #    y3 = np.concatenate((y, y2))

    #    l2f.update_task(X2, y2, task_id=0)

    #   assert np.array_equiv(l2f.task_id_to_X[0], X3)
    #    assert np.array_equiv(l2f.task_id_to_y[0], y3)


class TestUncertaintyForest:
    def test_initialize(self):
        uf = UncertaintyForest()
        assert True

    def test_correct_default_transformer(self):
        uf = UncertaintyForest()
        assert uf.default_transformer_class == TreeClassificationTransformer

    def test_correct_default_voter(self):
        uf = UncertaintyForest()
        assert uf.default_voter_class == TreeClassificationVoter

    def test_correct_default_decider(self):
        uf = UncertaintyForest()
        assert uf.default_decider_class == SimpleArgmaxAverage

    def test_correct_default_kwargs(self):
        uf = UncertaintyForest()

        # transformer
        assert uf.default_transformer_kwargs == {}

        # voter
        assert len(uf.default_voter_kwargs) == 1
        assert "kappa" in list(uf.default_voter_kwargs.keys())
        assert uf.default_voter_kwargs["kappa"] == np.inf

        # decider
        assert uf.default_decider_kwargs == {}

    def test_correct_default_n_estimators(self):
        uf = UncertaintyForest()
        assert uf.default_n_estimators == 100

    def test_correct_true_initilization_finite_sample_correction(self):
        uf = UncertaintyForest(kappa=np.inf)
        assert uf.default_voter_kwargs == {"kappa": np.inf}

    def test_predict_without_fit(self):
        # Generate random data
        X = np.random.normal(0, 1, size=(100, 3))

        with pytest.raises(NotFittedError):
            uf = UncertaintyForest()
            uf.predict(X)

    def test_predict(self):
        np.random.seed(1)

        uf = UncertaintyForest()

        X = np.concatenate((np.zeros(100), np.ones(100))).reshape(-1, 1)
        y = np.concatenate((np.zeros(100), np.ones(100)))

        uf.fit(X, y)
        u1 = uf.predict(np.array([0]).reshape(1, -1))
        u2 = uf.predict(np.array([1]).reshape(1, -1))
        assert u1 != u2

        u1 = uf.predict(np.array([0]).reshape(1, -1))
        u2 = uf.predict(np.array([0]).reshape(1, -1))
        assert u1 == u2

    def test_predict_proba(self):
        np.random.seed(1)

        uf = UncertaintyForest()

        X = np.concatenate((np.zeros(100), np.ones(100))).reshape(-1, 1)
        y = np.concatenate((np.zeros(100), np.ones(100)))

        uf.fit(X, y)
        u1 = uf.predict_proba(np.array([0]).reshape(1, -1))
        u2 = uf.predict_proba(np.array([1]).reshape(1, -1))
        assert not np.array_equiv(u1, u2)

        u1 = uf.predict_proba(np.array([0]).reshape(1, -1))
        u2 = uf.predict_proba(np.array([0]).reshape(1, -1))
        assert np.array_equiv(u1, u2)
