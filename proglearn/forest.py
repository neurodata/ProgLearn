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
        The number of estimators used in the Lifelong Classification Forest
    default_tree_construction_proportion : int, default=0.67
        The proportions of the input data set aside to train each decision 
        tree. The remainder of the data is used to fill in voting posteriors.
        This is used if 'tree_construction_proportion' is not fed to add_task.
    default_finite_sample_correction : bool, default=False
        Boolean indicating whether this learner will have finite sample correction
        This is used if 'finite_sample_correction' is not fed to add_task.

    Methods
    ---
    add_task(X, y, task_id)
        adds a task with id task_id, given input data matrix X 
        and output data matrix y, to the Lifelong Classification Forest
    add_transformer(X, y, transformer_id)
        adds a transformer with id transformer_id, trained on given input data matrix, X 
        and output data matrix, y, to the Lifelong Classification Forest. Also  
        trains the voters and deciders from new transformer to previous tasks, and will
        train voters and deciders from this transformer to all new tasks.
    predict(X, task_id)
        predicts class labels under task_id for each example in input data X.
    predict_proba(X, task_id)
        estimates class posteriors under task_id for each example in input data X.
    """
    def __init__(self, n_estimators=100, default_tree_construction_proportion=0.67, 
                 default_finite_sample_correction=False):
        self.n_estimators = n_estimators
        self.default_tree_construction_proportion = default_tree_construction_proportion
        self.default_finite_sample_correction = default_finite_sample_correction
        self.pl = ProgressiveLearner(
            default_transformer_class=TreeClassificationTransformer,
            default_transformer_kwargs={},
            default_voter_class=TreeClassificationVoter,
            default_voter_kwargs={"finite_sample_correction": default_finite_sample_correction},
            default_decider_class=SimpleArgmaxAverage,
            default_decider_kwargs={},
        )

        
       
    def add_task(self, X, y, task_id=None, tree_construction_proportion=None, 
                 finite_sample_correction=None):
        """
        adds a task with id task_id, given input data matrix X 
        and output data matrix y, to the Lifelong Classification Forest
        
        Parameters
        ---
        X : ndarray
            The input data matrix.
        y : ndarray
            The output (response) data matrix.
        task_id : obj, default=None
            The id corresponding to the task being added.
        tree_construction_proportion : int, default=None
            The proportions of the input data set aside to train each decision 
            tree. The remainder of the data is used to fill in voting posteriors.
            The default is used if 'None' is provided.
        finite_sample_correction : bool, default=False
            Boolean indicating whether this learner will have finite sample correction
            The default is used if 'None' is provided.
        """
        if tree_construction_proportion is None:
            tree_construction_proportion = self.default_tree_construction_proportion
        if finite_sample_correction is None:
            finite_sample_correction = self.default_finite_sample_correction
            
        self.pl.add_task(
            X,
            y,
            task_id=task_id,
            transformer_voter_decider_split=[tree_construction_proportion, 1-tree_construction_proportion, 0],
            num_transformers=self.n_estimators,
            transformer_kwargs = {"finite_sample_correction": finite_sample_correction},
            decider_kwargs = {"classes" : np.unique(y)}
        )
        return self

    def add_transformer(self, X, y, transformer_id=None):
        """
        adds a transformer with id transformer_id, trained on given input data matrix, X 
        and output data matrix, y, to the Lifelong Classification Forest. Also  
        trains the voters and deciders from new transformer to previous tasks, and will
        train voters and deciders from this transformer to all new tasks.
        
        Parameters
        ---
        X : ndarray
            The input data matrix.
        y : ndarray
            The output (response) data matrix.
        transformer_id : obj, default=None
            The id corresponding to the transformer being added.
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
        predicts class labels under task_id for each example in input data X.
        
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
        estimates class posteriors under task_id for each example in input data X.
        
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
        The number of trees in the UncertaintyForest
    finite_sample_correction : bool
        Boolean indicating whether this learner 
        will use finite sample correction
        
    Methods
    ---
    fit(X, y)
        fits forest to data X with labels y
    predict(X)
        predicts class labels for each example in input data X.
    predict_proba(X)
        estimates class posteriors for each example in input data X.
    """
    def __init__(self, n_estimators=100, finite_sample_correction=False):
        self.n_estimators = n_estimators
        self.finite_sample_correction = finite_sample_correction

    def fit(self, X, y):
        """
        fits forest to data X with labels y

        Parameters
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
        predicts class labels for each example in input data X.

        Parameters
        ---
        X : array of shape [n_samples, n_features]
            The data on which we are performing inference.
        """
        return self.lf.predict(X, 0)

    def predict_proba(self, X):
        """
        estimates class posteriors for each example in input data X.

        Parameters
        ---
        X : array of shape [n_samples, n_features]
            The data whose posteriors we are estimating.
        """
        return self.lf.predict_proba(X, 0)
