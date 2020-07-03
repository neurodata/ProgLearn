import abc
import numpy as np

class BaseDecider(abc.ABC):
    @abc.abstractmethod
    def fit(self, X, y):
        pass


    @abc.abstractmethod
    def predict(self, X):
        pass


class Average(BaseDecider):
    """
    Doc string here.
    """

    def __init__(self, transformer_ids='all', weights='simple'):
        self.transformer_ids = transformer_ids
        self.weights = weights
        self._is_fitted = False

    def fit(self, X, y=None):
        if self.weights is 'simple':
            self.weights = np.ones(np.shape(X)[0]) / np.shape(X)[0]
        else:
            pass

        self.decider = lambda x: np.average(x, axis=0, weights=self.weights)

    def predict(self, X):
        return np.argmax(self.decider(X), axis=1)