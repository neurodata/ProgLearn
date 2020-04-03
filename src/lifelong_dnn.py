from honest_dnn import HonestDNN 

from sklearn.base import clone 

import numpy as np

class LifeLongDNN():
    def __init__(self):
        self.X_across_tasks = []
        self.y_across_tasks = []
        
        self.transformers_across_tasks = []
        
        #element [i, j] votes on task i under transformation from task j
        self.voters_across_tasks_matrix = []
        self.num_tasks = 0
        
    def check_task_idx_(self, task_idx):
        if task_idx >= self.num_tasks:
            raise Exception("Invalid Task IDX")
    
    def add_task(self, X, y, epochs = 10):
        
        self.X_across_tasks.append(X)
        self.y_across_tasks.append(y)
        
        new_honest_dnn = HonestDNN()
        new_honest_dnn.fit(X, y, epochs = epochs)
        new_transformer = new_honest_dnn.get_transformer()
        new_voter = new_honest_dnn.get_voter()
        
        self.transformers_across_tasks.append(new_transformer)
        
        #add one voter to previous task voter lists under the new transformation
        for task_idx in range(self.num_tasks):            
            X_of_task, y_of_task = self.X_across_tasks[task_idx], self.y_across_tasks[task_idx]
            X_of_task_under_new_transform = new_transformer.predict(X_of_task) 
            
            unfit_task_voter_under_new_transformation = clone(self.voters_across_tasks_matrix[task_idx][0])
            task_voter_under_new_transformation = unfit_task_voter_under_new_transformation.fit(X_of_task_under_new_transform, y_of_task)

            self.voters_across_tasks_matrix[task_idx].append(task_voter_under_new_transformation)
            
        #add num_tasks voters to new task voter list under previous transformations 
        new_voters_under_previous_task_transformation = []
        for task_idx in range(self.num_tasks):
            transformer_of_task = self.transformers_across_tasks[task_idx]
            X_under_task_transformation = transformer_of_task.predict(X)
            
            unfit_new_task_voter_under_task_transformation = clone(new_voter)
            new_task_voter_under_task_transformation = unfit_new_task_voter_under_task_transformation.fit(X_under_task_transformation, y)
            new_voters_under_previous_task_transformation.append(new_task_voter_under_task_transformation)
            
        #make sure to add the voter of the new task under its own transformation
        new_voters_under_previous_task_transformation.append(new_voter)
        
        self.voters_across_tasks_matrix.append(new_voters_under_previous_task_transformation)
        
        self.num_tasks += 1
        
    def predict_proba(self, X, target_task_idx, transformer_task_idxs = None):
        self.check_task_idx_(target_task_idx)
        
        if transformer_task_idxs == None:
            transformer_task_idxs = range(self.num_tasks)
        
        posteriors_across_tasks = []
        for transformer_task_idx in transformer_task_idxs:
            transformer = self.transformers_across_tasks[transformer_task_idx]
            voter = self.voters_across_tasks_matrix[target_task_idx][transformer_task_idx]
            posteriors_across_tasks.append(voter.predict_proba(transformer.predict(X)))
        return np.mean(posteriors_across_tasks, axis = 0)
    
    def predict(self, X, target_task_idx, transformer_task_idxs = None):        
        return np.argmax(self.predict_proba(X, target_task_idx, transformer_task_idxs), axis = 1)
        