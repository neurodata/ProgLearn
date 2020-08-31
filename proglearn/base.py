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


class ClassificationDecider(BaseDecider):
    @abc.abstractmethod
    def predict_proba(self, X):
        pass
    
class BaseProgressiveLearner(abc.ABC):
    @abc.abstractmethod
    def add_task(self, X, y):
        pass

    @abc.abstractmethod
    def add_transformer(self, X, y):
        pass
    
    @abc.abstractmethod
    def predict(self, X, task_id):
        pass


class ClassificationProgressiveLearner(BaseProgressiveLearner):
    @abc.abstractmethod
    def predict_proba(self, X, task_id):
        pass
