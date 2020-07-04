import abc

class BaseTransformer(abc.ABC):
    """
    Doc strings here.
    """

    @abc.abstractmethod
    def fit(self, X=None, y=None):
        """
        Doc strings here.
        """

        pass

    @abc.abstractmethod
    def transform(self, X):
        """
        Doc strings here.
        """

        pass

    @abc.abstractmethod
    def is_fitted(self):
        """
        Doc strings here.
        """

        pass
    
class BaseVoter(abc.ABC):
    """
    Doc strings here.
    """

    @abc.abstractmethod
    def fit(self, X, y):
        """
        Doc strings here.
        """

        pass

    @abc.abstractmethod
    def vote(self, X):
        """
        Doc strings here.
        """

        pass

    @abc.abstractmethod
    def is_fitted(self):
        """
        Doc strings here.
        """
        
        pass

class BaseDecider(abc.ABC):
    @abc.abstractmethod
    def fit(self, X, y, transformer_id_to_transformers, voter_id_to_voters):
        pass


    @abc.abstractmethod
    def predict(self, X):
        pass