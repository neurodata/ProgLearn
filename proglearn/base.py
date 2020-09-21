'''
Main Author: Will LeVine 
Corresponding Email: levinewill@icloud.com
'''
from abc import ABC, abstractmethod

class BaseTransformer(ABC):
    """
    A base class for a transformer.
    
    Parameters
    ----------
    None
    """
    @abstractmethod
    def fit(self, X=None, y=None):
        """
        Fits the transformer.
        
        Parameters
        ----------
        X : ndarray
            Input data matrix. 
        y : ndarray
            Output (i.e. response) data matrix.
        """
        pass

    @abstractmethod
    def transform(self, X):
        """
        Perform inference using the transformer.
        
        Parameters
        ----------
        X : ndarray
            Input data matrix. 
        """
        pass

    @abstractmethod
    def is_fitted(self):
        """
        Indicates whether the transformer is fitted.
        
        Parameters
        ----------
        None
        """
        pass


class BaseVoter(ABC):
    """
    A base class for a voter.
    
    Parameters
    ----------
    None
    """
    @abstractmethod
    def fit(self, X, y):
        """
        Fits the voter.
        
        Parameters
        ----------
        X : ndarray
            Input data matrix. 
        y : ndarray
            Output (i.e. response) data matrix.
        """
        pass

    @abstractmethod
    def vote(self, X):
        """
        Perform inference using the voter.
        
        Parameters
        ----------
        X : ndarray
            Input data matrix. 
        """
        pass

    @abstractmethod
    def is_fitted(self):
        """
        Indicates whether the voter is fitted.
        
        Parameters
        ----------
        None
        """
        pass


class BaseDecider(ABC):
    """
    A base class for a decider.
    
    Parameters
    ----------
    None
    """
    @abstractmethod
    def fit(self, X, y, transformer_id_to_transformers, voter_id_to_voters):
        """
        Fits the decider.
        
        Parameters
        ----------
        X : ndarray
            Input data matrix. 
        y : ndarray
            Output (i.e. response) data matrix.
        transformer_id_to_transformers : dict
            A dictionary with keys of transformer ids and values of the corresponding transformers.
        voter_id_to_voters : dict
            A dictionary with keys of voter ids and values of the corresponding voter.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Perform inference using the decider.
        
        Parameters
        ----------
        X : ndarray
            Input data matrix. 
        """
        pass
    
    @abstractmethod
    def is_fitted(self):
        """
        Indicates whether the decider is fitted.
        
        Parameters
        ----------
        None
        """
        pass


class BaseClassificationDecider(BaseDecider):
    """
    A class for a decider which inherits from the base decider 
    but adds the functionality of estimating posteriors.
    
    Parameters
    ----------
    None
    """
    @abstractmethod
    def predict_proba(self, X):
        """
        Estimate posteriors using the decider.
        
        Parameters
        ----------
        X : ndarray
            Input data matrix. 
        """
        pass
    
class BaseProgressiveLearner(ABC):
    """
    A base class for a progressive learner.
    
    Parameters
    ----------
    None
    """
    @abstractmethod
    def add_task(self, X, y):
        """
        Add a new unseen task to the progressive learner.
        
        Parameters
        ----------
        X : ndarray
            Input data matrix. 
        y : ndarray
            Output (i.e. response) data matrix.
        """
        pass

    @abstractmethod
    def add_transformer(self, X, y):
        """
        Add a new transformer (but no voters or transformers corresponding to the task 
        from which the transformer data was collected.
        
        Parameters
        ----------
        X : ndarray
            Input data matrix. 
        y : ndarray
            Output (i.e. response) data matrix.
        """
        pass
    
    @abstractmethod
    def predict(self, X, task_id):
        """
        Perform inference corresponding to the input task_id using the progressive learner.
        
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        task_id : obj
            The task on which you are interested in performing inference.
        """
        pass


class BaseClassificationProgressiveLearner(BaseProgressiveLearner):
    """
    A class for a progressive learner which inherits from the base progressive learner 
    but adds the functionality of estimating posteriors for a given task_id.
    
    Parameters
    ----------
    None
    """
    @abstractmethod
    def predict_proba(self, X, task_id):
        """
        Estimate posteriors under a given task_id using the decider.
        
        Parameters
        ----------
        X : ndarray
            Input data matrix. 
        task_id : obj
            The task on which you are interested in estimating posteriors.
        """
        pass
