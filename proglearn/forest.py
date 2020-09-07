from .progressive_learner import ProgressiveLearner
from .transformers import TreeClassificationTransformer
from .voters import TreeClassificationVoter
from .deciders import SimpleAverage


class LifelongClassificationForest:
    def __init__(self, n_estimators=100, finite_sample_correction=False):
        self.n_estimators = n_estimators
        self.pl = ProgressiveLearner(
            default_transformer_class=TreeClassificationTransformer,
            default_transformer_kwargs={},
            default_voter_class=TreeClassificationVoter,
            default_voter_kwargs={"finite_sample_correction": finite_sample_correction},
            default_decider_class=SimpleAverage,
            default_decider_kwargs={},
        )

    def add_task(
        self, X, y, task_id=None, transformer_voter_decider_split=[0.67, 0.33, 0]
    ):
        self.pl.add_task(
            X,
            y,
            task_id=task_id,
            transformer_voter_decider_split=transformer_voter_decider_split,
            num_transformers=self.n_estimators,
        )
        return self

    def predict(self, X, task_id, transformer_ids=None):
        return self.pl.predict(X, task_id, transformer_ids=transformer_ids)

    def predict_proba(self, X, task_id, transformer_ids=None):
        return self.pl.predict_proba(X, task_id, transformer_ids=transformer_ids)


class UncertaintyForest:
    def __init__(self, n_estimators=100, finite_sample_correction=False):
        self.n_estimators = n_estimators
        self.finite_sample_correction = finite_sample_correction

    def fit(self, X, y):
        self.lf = LifelongClassificationForest(
            n_estimators=self.n_estimators,
            finite_sample_correction=self.finite_sample_correction,
        )
        self.lf.add_task(X, y, task_id=0)
        return self

    def predict(self, X):
        return self.lf.predict(X, 0)

    def predict_proba(self, X):
        return self.lf.predict_proba(X, 0)


class TransferForest:
    def __init__(self, n_estimators=100):
        self.lf = LifelongClassificationForest(n_estimators=n_estimators)
        self.source_ids = []

    def add_source_task(self, X, y, task_id=None):
        self.lf.add_task(
            X, y, task_id=task_id, transformer_voter_decider_split=[0.9, 0.1, 0]
        )
        self.source_ids.append(task_id)
        return self

    def add_target_task(self, X, y, task_id=None):
        self.lf.add_task(
            X, y, task_id=task_id, transformer_voter_decider_split=[0.1, 0.9, 0]
        )
        self.target_id = task_id
        return self

    def predict(self, X):
        return self.lf.predict(X, self.target_id, transformer_ids=self.source_ids)

    def predict_proba(self, X):
        return self.lf.predict_proba(X, self.target_id, transformer_ids=self.source_ids)