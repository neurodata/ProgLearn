import warnings

from sklearn.base import clone 
import numpy as np

from joblib import Parallel, delayed

# from deciders import *

import itertools

class ProgressiveLearner:
    def __init__(self, default_transformer_class = None, default_transformer_kwargs = {},
        default_voter_class = None, default_voter_kwargs = {},
        default_decider_class = None, default_decider_kwargs = {}
    ):
        """
        Doc strings here.
        """
        
        self.X_by_task_id = {}
        self.y_by_task_id = {}

        self.transformer_id_to_transformer = {} # transformer id to a fitted transformers
        self.task_id_to_transformer_id_to_voter = {} # task id to a map from transformer ids to a fitted voter
        self.task_id_to_deciders = {} # task id to a fitted decider 
        
        self.transformer_id_to_voter_class = {} # might be expensive to keep around and hints at need for default voters
        self.transformer_id_to_voter_kwargs = {}

        self.task_id_to_decider_class = {} # task id to uninstantiated decider class
        self.task_id_to_decider_kwargs = {} # task id to decider kwargs 

        self.default_transformer_class = default_transformer_class
        self.default_transformer_kwargs = default_transformer_kwargs

        self.default_voter_class = default_voter_class
        self.default_voter_kwargs = default_voter_kwargs

        self.default_decider_class = default_decider_class
        self.default_decider_kwargs = default_decider_kwargs
        
    def get_transformer_ids(self):
        """
        Doc strings here.
        """
        return np.array(list(self.transformer_id_to_transformer.keys()))

    def get_task_ids(self):
        """
        Doc strings here.
        """
        return np.array(list(self.task_id_to_deciders.keys()))
    
    def _append_transformer(self, transformer_id, transformer):
        if transformer_id in self.get_transformer_ids():
            self.transformer_id_to_transformer[transformer_id].append(transformer)
        else:
            self.transformer_id_to_transformer[transformer_id] = [transformer]
            
    def _append_voter(self, transformer_id, task_id, voter):
        if transformer_id in self.get_transformer_ids() and task_id in self.get_task_ids():
            self.task_id_to_transformer_id_to_voter[task_id][transformer_id].append(voter)
        elif task_id not in self.get_task_ids():
            self.task_id_to_transformer_id_to_voter[task_id] = {transformer_id: [voter]}
        else:
            self.task_id_to_transformer_id_to_voter[task_id][transformer_id] = [voter]
            
    def _get_transformer_voter_decider_idx(self, n, transformer_voter_decider_split):
        if transformer_voter_decider_split is None:
            transformer_idx, voter_idx, decider_idx = np.arange(n), np.arange(n), np.arange(n)
        elif np.sum(transformer_voter_decider_split) > 1:
            transformer_idx, voter_idx, decider_idx = [np.random.choice(np.arange(n), int(n*split)) for split in transformer_voter_decider_split]
        else:
            transformer_idx = np.random.choice(
                np.arange(n), 
                int(n*transformer_voter_decider_split[0])
            )

            voter_idx = np.random.choice(
                np.delete(np.arange(n), transformer_idx), 
                int(n*transformer_voter_decider_split[1])
            )

            decider_idx = np.random.choice(
                np.delete(np.arange(n), np.concatenate((transformer_idx, voter_idx))),
                int(n*transformer_voter_decider_split[2])
            )
        return transformer_idx, voter_idx, decider_idx

    def set_transformer(self, 
        X=None, y=None, transformer_id=None, transformer_class=None, transformer_kwargs={}, default_voter_class = None, default_voter_kwargs = {},
        transformer = None, acorn=None):
        """
        Doc string here.
        """

        # Keyword argument checking / default behavior

        transformer_ids = self.get_transformer_ids()
        if transformer_id is None:
            transformer_id = len(transformer_ids)

        if X is None and y is None:
            if transformer.is_fitted(): # need to define CHECK_IF_FITTED method
                self._append_transformer(transformer_id, transformer)
            else:
                raise ValueError('transformer_class is not fitted and X is None and y is None.')
            return 

        # Type check X

        if transformer_class is None:
            if self.default_transformer_class is None:
                raise ValueError("transformer_class is None and 'default_transformer_class' is None.")
            else:
                transformer_class = self.default_transformer_class

        if acorn is not None:
            np.random.seed(acorn)

        # Fit transformer and new voter
        if y is None:
            self._append_transformer(transformer_id, transformer_class(**transformer_kwargs).fit(X))
        else:
            # Type check y
            self._append_transformer(transformer_id, transformer_class(**transformer_kwargs).fit(X, y))
            
        self.transformer_id_to_voter_class[transformer_id] = default_voter_class
        self.transformer_id_to_voter_kwargs[transformer_id] = default_voter_kwargs

    def set_voter(self, X, y, transformer_id, task_id = None,
         voter_class = None, voter_kwargs = {}, bag_id = None):

        # Type check X

        # Type check y
        
        if task_id is None:
            task_id = len(self.get_task_ids())
            
        if voter_class is None:
            if self.transformer_id_to_voter_class[transformer_id] is not None:
                voter_class = self.transformer_id_to_voter_class[transformer_id]
            elif self.default_voter_class is not None:
                voter_class = self.default_voter_class
            else:
                raise ValueError("voter_class is None, the default voter class for the overall learner is None, and the default voter class for this transformer is None.")

        if voter_kwargs is None:
            if self.transformer_id_to_voter_kwargs[transformer_id] is not None:
                voter_kwargs = self.transformer_id_to_voter_kwargs[transformer_id]
            elif self.default_voter_kwargs is not None:
                voter_kwargs = self.default_voter_kwargs
            else:
                raise ValueError("voter_kwargs is None, the default voter kwargs for the overall learner is None, and the default voter kwargs for this transformer is None.")
        
        if bag_id == None:
            transformers = self.transformer_id_to_transformer[transformer_id]
        else:
            transformers = [bag_id]
        for _, transformer in enumerate(transformers):
            self._append_voter(transformer_id, task_id, voter_class(**voter_kwargs).fit(transformer.transform(X), y))

    def set_decider(self, task_id, transformer_ids,
        decider_class=None, decider_kwargs={}, X=None, y=None
        ):

        if decider_class is None:
            if self.default_decider_class is None:
                raise ValueError("decider_class is None and 'default_decider_class' is None.")
            else:
                decider_class = self.default_decider_class

        if decider_kwargs is None:
            if self.default_decider_kwargs is None:
                raise ValueError("decider_kwargs is None and 'default_decider_kwargs' is None.")
            else:
                decider_class = self.default_decider_kwargs
                
        # transformer_dict = {transformer_id : self.transformer_id_to_transformer[transformer_id] for transformer_id in transformer_ids}
        # voter_dict = {transformer_id : self.task_id_to_transformer_id_to_voter[transformer_id][task_id] for transformer_id in transformer_ids}

        # self.task_id_to_deciders[task_id] = decider_class(**decider_kwargs).fit(transformer_dict, voter_dict, X, y)
        
        self.task_id_to_deciders[task_id] = decider_class(**decider_kwargs).fit(X, y)

        self.task_id_to_decider_class[task_id] = decider_class
        self.task_id_to_decider_kwargs[task_id] = decider_kwargs
        
    def add_task(self, X, y, task_id=None, transformer_voter_decider_split = None, num_transformers = 1,
        tranformer_class=None, transformer_kwargs={}, 
        voter_class=None, voter_kwargs={}, 
        decider_class=None, decider_kwargs={},
        acorn=None):
        """
        Doc strings here.
        """
    
        # Type check X

        # Type check y
        
        self.X_by_task_id[task_id] = X
        self.y_by_yask_id[task_id] = y

        # Type check transformer_voter_decider_split
        n = np.shape(X)[0]
        
        # Keyword argument checking / default behavior

        if task_id is None:
            task_id = max(len(self.get_transformer_ids()), len(self.get_task_ids())) #come up with something that has fewer collisions

        if transformer_class is None and train_transformer:
            if self.default_transformer_class is None:
                raise ValueError("transformer_class is None and 'default_transformer_class' is None.")
            else:
                transformer_class = self.default_transformer_class

        if voter_class is None:
            if self.default_voter_class is None:
                raise ValueError("voter_class is None and 'default_voter_class' is None.")
            else:
                voter_class = self.default_voter_class

        if decider_class is None:
            if self.default_decider_class is None:
                raise ValueError("decider_class is None and 'default_decider_class' is None.")
            else:
                decider_class = self.default_decider_class
                
        if transformer_kwargs is None and train_transformer:
            if self.default_transformer_kwargs is None:
                raise ValueError("transformer_kwargs is None and 'default_transformer_kwargs' is None.")
            else:
                transformer_kwargs = self.default_transformer_kwargs

        if voter_kwargs is None:
            if self.default_voter_kwargs is None:
                raise ValueError("voter_kwargs is None and 'default_voter_kwargs' is None.")
            else:
                voter_kwargs = self.default_voter_kwargs

        if decider_kwargs is None:
            if self.default_decider_kwargs is None:
                raise ValueError("decider_kwargs is None and 'default_decider_kwargs' is None.")
            else:
                decider_kwargs = self.default_decider_kwargs

        if acorn is not None:
            np.random.seed(acorn)
        
        for transformer_num in range(num_transformers):
            transformer_idx, voter_idx, decider_idx = self._get_transformer_voter_decider_idx(n, transformer_voter_decider_split)
                
            self.set_transformer(X[transformer_idx], y[transformer_idx], task_id, tranformer_class, transformer_kwargs, voter_class, voter_kwargs)
        
            # train voters from previous tasks to new task
            for transformer_id in self.get_transformer_ids():
                if transformer_id == task_id:
                    self.set_voter(X[voter_idx], y[voter_idx], transformer_id, task_id, voter_class, voter_kwargs, bag_id = transformer_num)
                else:
                    self.set_voter(X, y, transformer_id, task_id, bag_id = transformer_num)
            if transformer_num == num_transformers - 1:
                self.set_decider(task_id, self.get_transformer_ids(), decider_class, decider_kwargs, X[decider_idx], y[decider_idx])

            # train voters from new transformer to previous tasks
            if num_transformers > 0:
                for existing_task_id in self.get_task_ids():
                    if existing_task_id == task_id:
                        continue
                    else:
                        existing_X = self.X_by_task_id[existing_task_id]
                        existing_y = self.y_by_task_id[existing_task_id]
                        self.set_voter(existing_X, existing_y, task_id, existing_task_id, bag_id = transformer_num)
                        if transformer_num == num_transformers - 1:
                            self.set_decider(existing_task_id, self.get_transformer_ids(), X = existing_X, y = existing_y)

        
    def predict(self, X, task_id):
        transformer_ids = self.task_id_to_decider[task_id].transformer_ids

        transformer_id_to_votes = {}
        for i, transformer_id in enumerate(transformer_ids):
            for transformer_num, transformer in enumerate(self.transformer_id_to_transformer[transformer_id]):
                X_transformed = transformer.transform(X)
                vote = self.task_id_to_transformer_id_to_voter[task_id][transformer_id][transformer_num].vote(X_transformed)
                if transformer_num == 0:
                    transformer_id_to_votes[transformer_id] = [vote]
                else:
                    transformer_id_to_votes[transformer_id].append(vote)
        return self.task_id_to_decider[task_id].predict(transformer_id_to_votes)
