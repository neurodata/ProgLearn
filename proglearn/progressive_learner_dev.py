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
        default_decider_class = None, default_decider_kwargs = {}):
        """
        Doc strings here.
        """
        
        self.X_by_task_id = {}
        self.y_by_yask_id = {}

        self.transformer_id_to_transformer = {} # transformer id to a fitted transformers
        self.task_id_to_voters = {} # task id to a map from transformer ids to a fitted voter
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

        # Instatiate classes to check for compatibility
        new_transformer = transformer_class(**transfer_kwargs)
        new_voter_for_new_tranformer = voter_class(**voter_kwargs)
        new_decider = decider_class(**decider_kwargs)

        # Fit transformer and new voter
        new_transformer.fit(X[transformer_idx], y[transformer_idx])
        transformed_X_voter = new_transformer.transform(X[voter_idx])
        new_voter_for_new_tranformer.fit(transformed_X_voter, y[voter_idx])

        new_transformer_id_to_voter = {}
        existing_transformer_id_to_voter = {}

        task_ids = self.get_task_ids()
        # Learning all new (transformer, voter) pairs
        for i, existing_transformer_id in enumerate(transformer_ids):
            # Transforming new data using old transformer and learning corresponding voter
            transformed_X = transformer_id_to_transformer[existing_task_id].transform(X)

            voter_class_for_existing_transformer = self.transformer_id_to_voter_class[existing_transformer_id]
            voter_kwargs_for_existing_transformer = self.transformer_id_to_voter_kwargs[existing_transformer_id]

            new_transformer_id_existing_voter[existing_transformer_id] = voter_class_for_existing_transformer(**voter_kwargs_for_existing_transformer).fit(transformed_X, y)
 
            # Transforming old data using new transformer and learning correspnding voter
            
            # Overwriting transformed X 
            if existing_transformer_id is in task_ids:
                transformed_X = new_transformer.transform(self.X_by_task_id[existing_task_id])
                existing_transformer_id_to_voter[existing_transformer_id] = voter_class(**voter_kwargs).fit(transformed_X, y)


        # Update deciders
        for i, existing_task_id in enumerate(task_ids):
            existing_decider_class = self.task_id_to_deciders[existing_task_id]
            existing_decider_kwargs = self.task_id_to_decider_kwargs[existing_task_id]

            # Get all the voters for the existing task id.
            existing_task_voters = 

            task_id_to_deciders = existing_decider_class.fit()

            pass

        # Update attributes
        self.X_by_task_id[task_id] = X
        self.y_by_task_id[task_id] = y




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



        # Train transformer
        # Train voters from previous transformers
        # 