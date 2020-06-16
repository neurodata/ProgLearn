import warnings

from sklearn.base import clone 
import numpy as np

from joblib import Parallel, delayed

import itertools

class ProgressiveLearner():
    def __init__(self, acorn = None, verbose = False):
        self.X_across_tasks = []
        self.y_across_tasks = []
        
        self.transformers_across_tasks = []
        
        #element [i, j] votes on decider from task i under representation from task j
        self.voters_across_tasks_matrix = []
        self.n_tasks = 0
        
        self.classes_across_tasks = []
        
        if acorn is not None:
            np.random.seed(acorn)
        
        self.verbose = verbose

        # Temporary attribute until Voters and Transformers are more uniform
        self.models = []


    def check_task_idx_(self, task_idx):
        if task_idx >= self.n_tasks:
            raise Exception("Invalid Task IDX")
    
    def add_model(self, 
                   X, 
                   y,
                   model='uf',
                   model_kwargs=None,
                   n_jobs=-1,
                   acorn = None):

        if acorn is not None:
            np.random.seed(acorn)

        self.models.append(model)
        
        # model_ is a temporary solution to the two voters/transformers not being uniform
        if model == "dnn":
            from .honest_dnn import HonestDNN as model_
        if model == "uf":
            from .uncertainty_forest import UncertaintyForest as model_
        
        self.X_across_tasks.append(X)
        self.y_across_tasks.append(y)

        if model_kwargs is None:
            model_kwargs = {}

        new_model = model_(**model_kwargs)
        new_model.fit(X, y, n_jobs=n_jobs)

        new_transformer = new_model.get_transformer(n_jobs=n_jobs)
        new_voter = new_model.get_voter()
        new_classes = new_model.classes_
        
        self.transformers_across_tasks.append(new_transformer)
        self.classes_across_tasks.append(new_classes)
        
        #add one voter to previous task voter lists under the new transformation
        for task_idx in range(self.n_tasks):
            X_of_task, y_of_task = self.X_across_tasks[task_idx], self.y_across_tasks[task_idx]
            if model == "dnn":
                X_of_task_under_new_transform = new_transformer.predict(X_of_task) 
            if model == "uf":
                X_of_task_under_new_transform = new_transformer(X_of_task) 

            unfit_task_voter_under_new_transformation = clone(self.voters_across_tasks_matrix[task_idx][0])
            if model == "uf":
                unfit_task_voter_under_new_transformation.classes_ = self.voters_across_tasks_matrix[task_idx][0].classes_

            task_voter_under_new_transformation = unfit_task_voter_under_new_transformation.fit(X_of_task_under_new_transform, y_of_task, n_jobs)

            self.voters_across_tasks_matrix[task_idx].append(task_voter_under_new_transformation)
            
        #add n_tasks voters to new task voter list under previous transformations 
        new_voters_under_previous_task_transformation = []
        for task_idx in range(self.n_tasks):
            other_task_model = self.models[task_idx]
            transformer_of_task = self.transformers_across_tasks[task_idx]

            if other_task_model == "dnn":
                X_under_task_transformation = transformer_of_task.predict(X)
            if other_task_model == "uf":
                X_under_task_transformation = transformer_of_task(X)

            unfit_new_task_voter_under_task_transformation = clone(new_voter)
            if other_task_model == "uf":
                unfit_new_task_voter_under_task_transformation.classes_ = new_voter.classes_

            new_task_voter_under_task_transformation = unfit_new_task_voter_under_task_transformation.fit(X_under_task_transformation, y, n_jobs)
            new_voters_under_previous_task_transformation.append(new_task_voter_under_task_transformation)
            
        #make sure to add the voter of the new task under its own transformation
        new_voters_under_previous_task_transformation.append(new_voter)
        
        self.voters_across_tasks_matrix.append(new_voters_under_previous_task_transformation)
        
        self.n_tasks += 1
        
    def _estimate_posteriors(self, X, representation = 0, decider = 0, n_jobs=-1):
        self.check_task_idx_(decider)

        print(n_jobs)
        
        if representation == "all":
            representation = range(self.n_tasks)
        elif isinstance(representation, int):
            representation = np.array([representation])

        model = self.models[decider]
        
        def worker(transformer_task_idx):
            transformer = self.transformers_across_tasks[transformer_task_idx]
            voter = self.voters_across_tasks_matrix[decider][transformer_task_idx]
            if model == "dnn":
                return voter.predict_proba(transformer.predict(X))
            if model == "uf":
                return voter.predict_proba(transformer(X), n_jobs=n_jobs)

        posteriors_across_tasks = np.array([worker(transformer_task_idx) for transformer_task_idx in representation])
        
        # if 
        # posteriors_across_tasks = np.array(
        #     Parallel(n_jobs=n_jobs)(
        #         delayed(worker)(transformer_task_idx) for transformer_task_idx in representation
        #     )
        # )    
        # else:
        #     posteriors_across_tasks = np.array([worker(transformer_task_idx) for transformer_task_idx in representation])    
            
        return posteriors_across_tasks

    def get_nonsimple_combination_weights(self, representation, decider, X_val, y_val, grid_step=0.01, n_jobs=-1):
        task_classes = self.classes_across_tasks[decider]

        posteriors_across_tasks_val = self._estimate_posteriors(X_val, representation, decider, n_jobs)

        if representation == "all":
            representation = np.arange(self.n_tasks)
        elif isinstance(representation, int):
            return np.array([1])

        J = len(representation)
        grid_1d = np.arange(0, 1+grid_step, step=grid_step)

        # Naively loop over ever elements of [0,1]^J
        n_correct_best=0
        for cart_prod in itertools.product(*(grid_1d for i in range(J))):
            # Only unique convex combinations
            if not np.isclose(np.sum(cart_prod), 1):
                continue

            temp_combined_posteriors_across_tasks_val = np.multiply(posteriors_across_tasks_val, cart_prod)
            temp_combined_posteriors_across_tasks_val = np.divide(
                                                            temp_combined_posteriors_across_tasks_val, 
                                                            np.sum(temp_combined_posteriors_across_tasks_val, axis=1)[:, np.newaxis]
                                                        )
            temp_predictions_val = task_classes[np.argmax(temp_combined_posteriors_across_tasks_val, axis = -1)]
            n_correct = np.sum(temp_predictions_val == y_val)

            if n_correct > n_correct_best:
                n_correct_best = n_correct
                cart_prod_best = cart_prod

        return cart_prod_best

        
    def predict(self, X, representation = 0, decider = 0, weights="simple", n_jobs=-1):
        task_classes = self.classes_across_tasks[decider]

        posteriors_across_tasks = self._estimate_posteriors(X, representation, decider, n_jobs)

        if weights == "simple":
            combined_posteriors_across_tasks = np.mean(posteriors_across_tasks, axis=0)
        elif isinstance(representation, int):
            combined_posteriors_across_tasks = np.mean(posteriors_across_tasks, axis=0)
        else:
            if representation == "all":
                representation = np.arange(self.n_tasks)

            if len(weights) != len(representation):
                raise ValueError('Length of weight vector must be the same as the number of representations used.')

            if np.sum(weights) == 0:
                warnings.warn("Weight vector sums to 0, using simple average.")
                combined_posteriors_across_tasks = np.mean(posteriors_across_tasks, axis=0)
            elif np.sum(weights < 0):
                raise ValueError("Negative weight.")
            else:
                weights = weights / np.sum(weights)
                combined_posteriors_across_tasks = np.average(posteriors_across_tasks, axis=0, weights=weights) 

        return task_classes[np.argmax(combined_posteriors_across_tasks, axis = -1)]