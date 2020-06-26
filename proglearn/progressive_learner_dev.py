import warnings

from sklearn.base import clone 
import numpy as np

from joblib import Parallel, delayed

from deciders import *
from transformers import *
from voters import *

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
        self.y_by_yask_id = {}

        self.transformer_id_to_transformer = {} # transformer id to a fitted transformers
        self.transformer_id_to_voters = {} # task id to a map from transformer ids to a fitted voter
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
        return np.array(transformer_id_to_transformer.keys())

    def get_task_ids(self):
        """
        Doc strings here.
        """
        return np.array(task_id_to_deciders.keys())

    def set_transformer(self, 
        X=None, y=None, transformer_id=None, tranformer_class=None, transformer_kwargs={}, default_voter_class = None, default_voter_kwargs = {}
        ):
        """
        Doc string here.
        """

        # Keyword argument checking / default behavior

        transformer_ids = self.get_transformer_ids()
        if transformer_id is None:
            transformer_id = len(transformer_ids)

        if X is None and y is None:
            CHECK_IF_FITTED(transformer_class) # need to define CHECK_IF_FITTED method
            self.transformer_id_to_transformer[transformer_id] = transformer_class
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
            self.transformer_id_to_transformer[transformer_id] = transformer_class(**transformer_kwargs).fit(X)
        else:
            # Type check y
            self.transformer_id_to_transformer[transformer_id] = transformer_class(**transformer_kwargs).fit(X, y)
            
        self.transformer_id_to_voter_class[transformer_id] = default_voter_class
        self.transformer_id_to_voter_kwargs[transformer_id] = default_voter_kwargs

    def set_voter(self, X, y, transformer_id, task_id = None,
         voter_class = None, voter_kwargs = {}):

        # Type check X

        # Type check y
        
        if task_id = None:
            task_id = len(self.get_task_ids())
            
        if voter_class is None:
            if self.transformer_id_to_voter_class[transformer_id] is not None:
                voter_class = self.transformer_id_to_voter_class[transformer_id]
            elif self.default_voter_class is not None:
                voter_class = self.default_voter_class
            else
                raise ValueError("voter_class is None, the default voter class for the overall learner is None, and the default voter class for this transformer is None.")

        if voter_kwargs is None:
            if self.transformer_id_to_voter_kwargs[transformer_id] is not None:
                voter_kwargs = self.transformer_id_to_voter_kwargs[transformer_id]
            elif self.default_voter_kwargs is not None:
                voter_kwargs = self.default_voter_kwargs
            else
                raise ValueError("voter_kwargs is None, the default voter kwargs for the overall learner is None, and the default voter kwargs for this transformer is None.")
        
        transformer = self.transformer_id_to_transformer[transformer_id]
        self.transformer_id_to_voters[transformer_id][task_id] = voter_class(**voter_kwargs).fit(transformer.transform(X), y)

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
                
        voters = [self.transformer_id_to_voters[transformer_id][task_id] for transformer_id in transformer_ids]
        transformers = [self.transformer_id_to_transformer[transformer_id] ffor transformer_id in transformer_ids]

        self.task_id_to_deciders[task_id] = decider_class(**decider_kwargs).fit(voters, transformers, X, y)
        
        self.task_id_to_decider_class[task_id] = decider_class
        self.task_id_to_decider_kwargs[task_id] = decider_kwargs
        
    def set_triple(self, X, y, transformer_voter_decider_split, task_id=None, 
        tranformer_class=None, transformer_kwargs={}, 
        voter_class=None, voter_kwargs={}, 
        decider_class=None, decider_kwargs={},
        acorn=None):
        """
        Doc strings here.
        """
    
        # Type check X

        # Type check y

        # Type check transformer_voter_decider_split

        # Assumes at least 2d array
        n = np.shape(X)[0]
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
                np.delete(np.arange(n), np.concatenate((transformer_idx, voter_idx)))
                int(n*transformer_voter_decider_split[2])
            )

        # Keyword argument checking / default behavior

        transformer_ids = self.get_transformer_ids()
        if task_id is None:
            task_id = len(transformer_ids)

        if transformer_class is None:
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

        if acorn is not None:
            np.random.seed(acorn)

        self.set_transformer(X[transformer_idx], y[transformer_idx], task_id, tranformer_class, transformer_kwargs)
        self.set_voter(task_id, X[voter_idx], y[voter_idx], voter_class, voter_kwargs, add_cross_task=True)

        for i, existing_task_id in enumerate(self.get_task_ids()):
            if existing_task_id == task_id:
                self.set_decider(existing_task_id, self.transformer_id_to_voters[existing_task_id], decider_class, decider_kwargs, 
                    transformers=self.transformer_transformer_id.values(), X=X[decider_idx], y=y[decider_idx])
            else:
                self.set_decider(existing_task_id, self.transformer_id_to_voters[existing_task_id], 
                    task_id_to_decider_class[existing_task_id], task_id_to_decider_kwargs[existing_task_id], 
                    transformers=self.transformer_transformer_id.values(), X=X_by_task_id[existing_task_id], y=y_by_task_id[existing_task_id])



        # # Instatiate classes to check for compatibility
        # new_transformer = transformer_class(**transfer_kwargs)
        # new_voter_for_new_tranformer = voter_class(**voter_kwargs)
        # new_decider = decider_class(**decider_kwargs)

        # # Fit transformer and new voter
        # new_transformer.fit(X[transformer_idx], y[transformer_idx])
        # transformed_X_voter = new_transformer.transform(X[voter_idx])
        # new_voter_for_new_tranformer.fit(transformed_X_voter, y[voter_idx])

        # transformer_id_to_voter_new_task_id = {}
        # existing_task_id_to_new_voter = {}

        # task_ids = self.get_task_ids()
        # # Learning all new (transformer, voter) pairs
        # for i, existing_transformer_id in enumerate(transformer_ids):
        #     # Transforming new data using old transformer and learning corresponding voter
        #     transformed_X = transformer_id_to_transformer[existing_task_id].transform(X)

        #     voter_class_for_existing_transformer = self.transformer_id_to_voter_class[existing_transformer_id]
        #     voter_kwargs_for_existing_transformer = self.transformer_id_to_voter_kwargs[existing_transformer_id]

        #     transformer_id_to_voter_new_task_id[existing_transformer_id] = voter_class_for_existing_transformer(**voter_kwargs_for_existing_transformer).fit(transformed_X, y)
 
        #     # Transforming old data using new transformer and learning correspnding voter
            
        #     # Overwriting transformed X 
        #     if existing_transformer_id is in task_ids:
        #         transformed_X = new_transformer.transform(self.X_by_task_id[existing_task_id])
        #         existing_task_id_to_new_voter[existing_transformer_id] = voter_class(**voter_kwargs).fit(transformed_X, y)

        # self.transformer_id_to_transformer[task_id] = new_transformer

        # # Fit new task decider
        # new_decider.fit(
        #     existing_transformer_id.values(), 
        #     transformers = transformer_id_to_transformer,
        #     X = X,
        #     y = y
        # )

        # # Update existing task deciders
        # updated_deciders = {task_id: new_decider}
        # for i, existing_task_id in enumerate(task_ids):
        #     existing_decider_class = self.task_id_to_deciders[existing_task_id]
        #     existing_decider_kwargs = self.task_id_to_decider_kwargs[existing_task_id]

        #     # Get all the voters for the existing task id.
        #     existing_task_voters = self.transformer_id_to_voters[existing_task_id].values()

        #     existing_task_decider_class = self.task_id_to_decider_class[existing_task_id]
        #     existing_task_decider_kwargs = self.task_id_to_decider_kwargs[existing_task_id]

        #     updated_deciders[existing_task_id] = existing_task_decider_class(**existing_decider_kwargs).fit(
        #                                             existing_task_voters,
        #                                             transformers = transformer_id_to_transformer,
        #                                             X = self.X_by_task_id[existing_task_id],
        #                                             y = self.y_by_task_id[existing_task_id]
        #                                         )

        # # Update attributes
        # self.X_by_task_id[task_id] = X
        # self.y_by_task_id[task_id] = y

        # self.transformer_id_to_voters[task_id] = transformer_id_to_voter_new_task_id
        # for i, existing_task_id in enumerate(task_ids):
        #     self.transformer_id_to_voters[existing_task_id][task_id] = existing_task_id_to_new_voter[existing_transformer_id]
        
        # self.task_id_to_deciders = updated_deciders 
        
        # self.transformer_id_to_voter_class[task_id] = voter_class # might be expensive to keep around and hints at need for default voters
        # self.transformer_id_to_voter_kwargs[task_id] = voter_kwargs

        # self.task_id_to_decider_class[task_id] = decider_class # task id to uninstantiated decider class
        # self.task_id_to_decider_kwargs[task_id] = decider_kwargs # task id to decider kwargs 
        
    def predict(self, X, task_id):
        raise Exception("predict unimplemented")
