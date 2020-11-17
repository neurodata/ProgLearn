"""
Main Author: Will LeVine 
Corresponding Email: levinewill@icloud.com
"""
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


class BaseTransformer(ABC, BaseEstimator, TransformerMixin):
    """
    A base class for a transformer, derived from scikit-learn's BaseEstimator
    class and TransformerMixin mixin.

    Parameters
    ----------
    None

    Attributes
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


class BaseVoter(ABC, BaseEstimator):
    """
    A base class for a voter, derived from scikit-learn's BaseEstimator class.

    Parameters
    ----------
    None

    Attributes
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
            Transformed data matrix.

        y : ndarray
            Output (i.e. response) data matrix.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Perform inference using the voter.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        pass


class BaseClassificationVoter(BaseVoter, ClassifierMixin):
    """
    A class for a voter which inherits from scikit-learn's ClassifierMixin
    mixin and the base voter, with the additional functionality of
    performing inference.

    Parameters
    ----------
    None

    Attributes
    ----------
    None
    """

    @abstractmethod
    def predict_proba(self, X):
        """
        Perform inference using the voter on transformed data X.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        pass


class BaseDecider(ABC, BaseEstimator):
    """
    A base class for a decider, derived from scikit-learn's BaseEstimator class.

    Parameters
    ----------
    None

    Attributes
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


class BaseClassificationDecider(BaseDecider, ClassifierMixin):
    """
    A class for a decider which inherits from scikit-learn's ClassifierMixin
    mixin and the base decider, with added functionality of estimating posteriors.

    Parameters
    ----------
    None

    Attributes
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

    Attributes
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
        from which the transformer data was collected).

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
        Perform inference corresponding to the input task_id on input data X using the
        progressive learner.

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

    Attributes
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
