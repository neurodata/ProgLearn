import abc

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

    def __init__(self, transformers, voters, weights='simple'):
        self._is_fitted = False
        self.weights = weights

    def fit(self, X=None, y=None, transformers=None, voters=None, X_is_votes=True):
        if self.weights == 'simple':
            if X_is_votes
            self.weights = np.ones(np.shape(voters)[0]) / np.shape(voters)[0]
        else:
            pass

        self.decider = lambda vs: np.average(vs, weights)

    def predict(self, X, transformers, voters):
        votes = []
        for i, transformer_id in enumerate(self.transformer_ids):
            X_transformed = transformers[transformer_id].transform(X)
            X_votes.append(voters[transformer_id].vote(X_transformed))
        votes = np.array(votes)

        return np.array([self.decider(vs) for vs in votes])