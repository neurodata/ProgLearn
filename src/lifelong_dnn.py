'''
Primary Author: Will LeVine
Email: levinewill@icloud.com
Latest Update: June 11, 2020
'''

from sklearn.base import clone 

import numpy as np

from joblib import Parallel, delayed

class LifeLongDNN():
    def __init__(self, acorn = None, verbose = False, model = "uf", parallel = True, n_jobs = None):
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
        
        self.model = model
        
        self.parallel = parallel
        
        self.n_jobs = n_jobs
        
    def check_task_idx_(self, task_idx):
        if task_idx >= self.n_tasks:
            raise Exception("Invalid Task IDX")
    
    def new_forest(self, 
                   X, 
                   y, 
                   epochs = 100, 
                   lr = 5e-4, 
                   n_estimators = 100, 
                   max_samples = .63,
                   bootstrap = False,
                   max_depth = 30,
                   min_samples_leaf = 1,
                   acorn = None,
                   parallel = False,
                   n_jobs = None):
        
        if self.model == "dnn":
            from honest_dnn import HonestDNN 
        if self.model == "uf":
            from uncertainty_forest import UncertaintyForest
        
        self.X_across_tasks.append(X)
        self.y_across_tasks.append(y)
        
        if self.model == "dnn":
            new_honest_dnn = HonestDNN(verbose = self.verbose)
            new_honest_dnn.fit(X, y, epochs = epochs, lr = lr)
        if self.model == "uf":
            new_honest_dnn = UncertaintyForest(n_estimators = n_estimators,
                                               max_samples = max_samples,
                                               bootstrap = bootstrap,
                                               max_depth = max_depth,
                                               min_samples_leaf = min_samples_leaf,
                                               parallel = parallel,
                                               n_jobs = n_jobs)
            new_honest_dnn.fit(X, y)
        new_transformer = new_honest_dnn.get_transformer()
        new_voter = new_honest_dnn.get_voter()
        new_classes = new_honest_dnn.classes_
        
        self.transformers_across_tasks.append(new_transformer)
        self.classes_across_tasks.append(new_classes)
        
        #add one voter to previous task voter lists under the new transformation
        for task_idx in range(self.n_tasks):
            X_of_task, y_of_task = self.X_across_tasks[task_idx], self.y_across_tasks[task_idx]
            if self.model == "dnn":
                X_of_task_under_new_transform = new_transformer.predict(X_of_task) 
            if self.model == "uf":
                X_of_task_under_new_transform = new_transformer(X_of_task) 
            unfit_task_voter_under_new_transformation = clone(self.voters_across_tasks_matrix[task_idx][0])
            if self.model == "uf":
                unfit_task_voter_under_new_transformation.classes_ = self.voters_across_tasks_matrix[task_idx][0].classes_
            task_voter_under_new_transformation = unfit_task_voter_under_new_transformation.fit(X_of_task_under_new_transform, y_of_task)

            self.voters_across_tasks_matrix[task_idx].append(task_voter_under_new_transformation)
            
        #add n_tasks voters to new task voter list under previous transformations 
        new_voters_under_previous_task_transformation = []
        for task_idx in range(self.n_tasks):
            transformer_of_task = self.transformers_across_tasks[task_idx]
            if self.model == "dnn":
                X_under_task_transformation = transformer_of_task.predict(X)
            if self.model == "uf":
                X_under_task_transformation = transformer_of_task(X)
            unfit_new_task_voter_under_task_transformation = clone(new_voter)
            if self.model == "uf":
                unfit_new_task_voter_under_task_transformation.classes_ = new_voter.classes_
            new_task_voter_under_task_transformation = unfit_new_task_voter_under_task_transformation.fit(X_under_task_transformation, y)
            new_voters_under_previous_task_transformation.append(new_task_voter_under_task_transformation)
            
        #make sure to add the voter of the new task under its own transformation
        new_voters_under_previous_task_transformation.append(new_voter)
        
        self.voters_across_tasks_matrix.append(new_voters_under_previous_task_transformation)
        
        self.n_tasks += 1
        
    def _estimate_posteriors(self, X, representation = 0, decider = 0):
        self.check_task_idx_(decider)
        
        if representation == "all":
            representation = range(self.n_tasks)
        elif isinstance(representation, int):
            representation = np.array([representation])
        
        def worker(transformer_task_idx):
            transformer = self.transformers_across_tasks[transformer_task_idx]
            voter = self.voters_across_tasks_matrix[decider][transformer_task_idx]
            if self.model == "dnn":
                return voter.predict_proba(transformer.predict(X))
            if self.model == "uf":
                return voter.predict_proba(transformer(X))
        
        if self.parallel:
            posteriors_across_tasks = np.array(
                        Parallel(n_jobs=self.n_jobs if self.n_jobs != None else len(representation))(
                                delayed(worker)(transformer_task_idx) for transformer_task_idx in representation
                        )
                )    
        else:
            posteriors_across_tasks = np.array([worker(transformer_task_idx) for transformer_task_idx in representation])    
            
        return np.mean(posteriors_across_tasks, axis = 0)
        
    def predict(self, X, representation = 0, decider = 0):
        task_classes = self.classes_across_tasks[decider]
        return task_classes[np.argmax(self._estimate_posteriors(X, representation, decider), axis = -1)]
        