import numpy as np

from base import BaseDecider


class SimpleAverage(BaseDecider):
    """
    Doc string here.
    """

    def __init__(self, classes = []):
        self.classes = classes

    def fit(self, y, transformer_id_to_transformers, classes = None, transformer_id_to_voters = None, X=None):
        self.classes = self.classes if len(self.classes) > 0 else np.unique(y)
        self.transformer_id_to_transformers = transformer_id_to_transformers
        self.transformer_id_to_voters = transformer_id_to_voters
        return self

    def predict(self, X, transformer_ids = None):
        vote_per_transformer_id = []
        for transformer_id in transformer_ids if transformer_ids else self.transformer_id_to_voters.keys():
            vote_per_bag_id = []
            for bag_id in range(len(self.transformer_id_to_transformers[transformer_id])):
                transformer = self.transformer_id_to_transformers[transformer_id][bag_id]
                X_transformed = transformer.transform(X)
                voter = self.transformer_id_to_voters[transformer_id][bag_id]
                vote = voter.vote(X_transformed)
                vote_per_bag_id.append(vote)
            vote_per_transformer_id.append(np.mean(vote_per_bag_id, axis = 0))
        vote_overall = np.mean(vote_per_transformer_id, axis = 0)
        return self.classes[np.argmax(vote_overall, axis=1)]

class KNNRegressionDecider(BaseDecider):
    """
    Doc string here.
    """
    def __init__(self, k = None, transformer_id_to_transformers = None, transformer_id_to_voters = None):
        self.k = k

        # Dict that maps transformer_id (int) to a Transformer object.
        self.transformer_id_to_transformers = transformer_id_to_transformers
        # Dict that maps transformer_id (int) to a Voter object (fitted on the output a Transformer).
        self.transformer_id_to_voters = transformer_id_to_voters
        self._is_fitted = False
        
    def fit(self, X, y):
        if not self.k:
            self.k = 16 * int(np.log2(len(X)))

        # Form yhats, n-by-num_transformers matrix that act as the "new X".
        # Here, n is the training set size of the particular task that this is deciding on.
        num_transformers = len(self.transformer_id_to_transformers.keys())
        yhats = np.zeros((n, num_transformers))

        for transformer_id in self.transformer_id_to_voters:
           
            # The zero index is for the 'bag_id' as in random forest,
            # where multiple transformers are bagged together in each hypothesis.
            transformer = self.transformer_id_to_transformers[transformer_id][0]
            X_transformed = transformer.transform(X)
            voter = self.transformer_id_to_voters[transformer_id][0]
            yhats[:, transformer_id] = voter.vote(X_transformed)

        self.knn = KNeighborsRegressor(self.k, weights = "distance", p = 1)
        self.knn.fit(yhats, y)

        self._is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted():
            msg = (
                    "This %(name)s instance is not fitted yet. Call 'fit' with "
                    "appropriate arguments before using this transformer."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})
        
        X = check_array(X)
        
        transformer_ids = self.transformer_id_to_transformers.keys()
        yhats = np.zeros((n, len(transformer_ids)))

        # Transformer IDs may not necessarily be {0, ..., num_transformers - 1}.
        for i in range(len(transformer_ids)):
           
            transformer_id = transformer_ids[i]

            # The zero index is for the 'bag_id' as in random forest,
            # where multiple transformers are bagged together in each hypothesis.
            transformer = self.transformer_id_to_transformers[transformer_id][0]
            X_transformed = transformer.transform(X)
            voter = self.transformer_id_to_voters[transformer_id][0]
            yhats[:, i] = voter.vote(X_transformed)

        return self.knn.predict(yhats)

class NeuralRegressionDecider(BaseDecider):
    """
    Doc string here.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
        