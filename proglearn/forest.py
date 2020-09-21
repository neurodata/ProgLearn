'''
Main Author: Will LeVine
Corresponding Email: levinewill@icloud.com
'''
from .progressive_learner import ProgressiveLearner
from .transformers import TreeClassificationTransformer
from .voters import TreeClassificationVoter
from .deciders import SimpleArgmaxAverage
import numpy as np

class LifelongClassificationForest:
    """
    A class used to represent a lifelong classification forest.
    
    Parameters:
    ---
    n_estimators : int, default=100
        The number of estimators used in the LifelongClassificationForest
    tree_construction_proportion : int, default=0.67
        The proportions of the input data set aside to train each decision 
        tree. The remainder of the data is used to train the transformers. 
        This is used in case you are using a bagging algorithm ensemble 
        to train on disjoint subsets of the data. This parameter is mostly 
        for internal use.
    finite_sample_correction : bool, default=False
        Boolean indicating whether this learner will have finite sample correction 

    Methods
    ---
    add_task(X, y, task_id)
        adds a task with id task_id, given input data matrix X 
        and output data matrix y, to the progressive learner
    add_transformer(X, y, transformer_id)
        adds a transformer with id transformer_id, given input data matrix, X 
        and output data matrix, y, to the progressive learner and trains the 
        voters and deciders from new transformer to previous tasks
    predict(X, task_id)
        predicts class labels of a task with id task_id given 
        input data matrix X 
    predict_proba(X, task_id)
        predicts posterior probabilities of each class label of a task 
        with id task_id given input data matrix X and 
    """
    def __init__(self, n_estimators=100, tree_construction_proportion=0.67, finite_sample_correction=False):
        self.n_estimators = n_estimators
        self.tree_construction_proportion=tree_construction_proportion
        self.pl = ProgressiveLearner(
            default_transformer_class=TreeClassificationTransformer,
            default_transformer_kwargs={},
            default_voter_class=TreeClassificationVoter,
            default_voter_kwargs={"finite_sample_correction": finite_sample_correction},
            default_decider_class=SimpleArgmaxAverage,
            default_decider_kwargs={},
        )

        
       
    def add_task(
        self, X, y, task_id=None):
        """
        adds a task with id task_id, given input data matrix X 
        and output data matrix y, to the progressive learner
        
        Parameters
        ---
        X : ndarray
            The input data matrix.
        y : ndarray
            The output (response) data matrix.
        task_id : obj, default=None
            The id corresponding to the task being added.
        
        Attributes
        ---
        tree_construction_proportion : float
            The proportions of the input data set aside to 
            train each decision tree
        n_estimators : int
            The number of estimators used in the LifelongClassificationForest
        """
        self.pl.add_task(
            X,
            y,
            task_id=task_id,
            transformer_voter_decider_split=[self.tree_construction_proportion, 1-self.tree_construction_proportion, 0],
            num_transformers=self.n_estimators,
            decider_kwargs = {"classes" : np.unique(y)}
        )
        return self

    def add_transformer(self, X, y, transformer_id=None):
        """
        adds a transformer to the progressive learner and trains the voters 
        and deciders from this new transformer to previous tasks
        
        Parameters
        ---
        X : ndarray
            The input data matrix.
        y : ndarray
            The output (response) data matrix.
        transformer_id : obj, default=None
            The id corresponding to the transformer being added.

        Attributes
        ---
        tree_construction_proportion : float
            The proportions of the input data set aside to 
            train each decision tree
        n_estimators : int
            The number of estimators used in the LifelongClassificationForest
        """
        self.pl.add_transformer(
            X,
            y,
            transformer_id=transformer_id,
            num_transformers=self.n_estimators,
            transformer_data_proportion=self.tree_construction_proportion
        )

        return self

    def predict(self, X, task_id):
        """
        predicts the class labels for a particular task 
        given data, X and task id, task_id
        
        Parameters 
        ---
        X : ndarray
            The input data matrix.
        task_id : obj
            The id corresponding to the task being mapped to.
        """
        return self.pl.predict(X, task_id)

    def predict_proba(self, X, task_id):
        """
        predicts the posterior probabilities of each class for 
        a particular task given data, X and task id, task_id
        
        Parameters
        ---
        X : ndarray
            The input data matrix.
        task_id:
            The id corresponding to the task being mapped to.
        """
        return self.pl.predict_proba(X, task_id)


class UncertaintyForest:
    """
    A class used to represent an uncertainty forest.

    Attributes
    ---
    lf : LifelongClassificationForest
        A lifelong classification forest object
    n_estimators : int
        The number of estimaters used in the 
        LifelongClassificationForest
    finite_sample_correction : bool
        Boolean indicating whether this learner 
        will have finite sample correction used
        as in LifelongClassificationForest
        
    Methods
    ---
    fit(X, y)
        fits forest to data X with labels y
    predict(X)
        predicts class labels given data, X
    predict_proba(X)
        predicts posterior probabilities given data, X, of each class label
    """
    def __init__(self, n_estimators=100, finite_sample_correction=False):
        self.n_estimators = n_estimators
        self.finite_sample_correction = finite_sample_correction

    def fit(self, X, y):
        """
        fits data X given class labels y

        Attributes
        ---
        X : array of shape [n_samples, n_features]
            The data that will be trained on
        y : array of shape [n_samples]
            The label for cluster membership of the given data
        """
        self.lf = LifelongClassificationForest(
            n_estimators = self.n_estimators,
            finite_sample_correction = self.finite_sample_correction
        )
        self.lf.add_task(X, y, task_id=0)
        return self

    def predict(self, X):
        """
        predicts the class labels given data X

        Attributes
        ---
        X : array of shape [n_samples, n_features]
            The data on which we are performing inference.
        """
        return self.lf.predict(X, 0)

    def predict_proba(self, X):
        """
        returns the posterior probabilities of each class for data X

        Attributes
        ---
        X : array of shape [n_samples, n_features]
            The data whose posteriors we are estimating.
        """
        return self.lf.predict_proba(X, 0)
