from .progressive_learner import ProgressiveLearner
from .transformers import TreeClassificationTransformer
from .voters import TreeClassificationVoter
from .deciders import SimpleAverage

from sklearn.utils import check_X_y, check_array


class LifelongForest():
    
    def __init__(self, n_estimators=100, finite_sample_correction=False):
        self.n_estimators = n_estimators
        self.pl = ProgressiveLearner(default_transformer_class=
                                     TreeClassificationTransformer, 
                                     default_transformer_kwargs={},
                                     default_voter_class=TreeClassificationVoter,
                                     default_voter_kwargs=
                                     {'finite_sample_correction' : 
                                      finite_sample_correction}, 
                                     default_decider_class=SimpleAverage,
                                     default_decider_kwargs={})
        
    def add_task(self, X, y, task_id=None):
        X, y = check_X_y(X, y)
        self.pl.add_task(X, 
                         y, 
                         task_id=task_id, 
                         transformer_voter_decider_split=[0.67, 0.33, 0], 
                         num_transformers=self.n_estimators)
        
    def predict(self, X, task_id):
        X = check_array(X)
        return self.pl.predict(X, task_id)


class UncertaintyForest():
    
    def __init__(self, n_estimators=100, finite_sample_correction=False):
        self.n_estimators = n_estimators
        self.finite_sample_correction = finite_sample_correction
    
    def fit(self, X, y):
        self.lf = LifelongForest(n_estimators=self.n_estimators,
                                 finite_sample_correction=self.finite_sample_correction)
        self.lf.add_task(X, y, task_id=0)
        
    def predict(self, X):
        return self.lf.predict(X, 0)
