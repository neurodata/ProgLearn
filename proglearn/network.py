'''
Main Author: Will LeVine 
Corresponding Email: levinewill@icloud.com
'''
import numpy as np

from .progressive_learner import ProgressiveLearner
from .transformers import NeuralClassificationTransformer, NeuralRegressionTransformer
from .voters import KNNClassificationVoter, NeuralRegressionVoter
from .deciders import SimpleAverage, LinearRegressionDecider, KNNRegressionDecider

from sklearn.utils import check_X_y, check_array
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

class LifelongClassificationNetwork:
    def __init__(
        self,
        network,
        loss="categorical_crossentropy",
        epochs=100,
        lr=3e-4,
        batch_size=32,
        verbose=False,
    ):
        self.network = network
        self.loss = loss
        self.epochs = epochs
        self.optimizer = Adam(lr)
        self.verbose = verbose
        self.batch_size = batch_size
        self.is_first_task = True

    def setup(self, num_points_per_task):

        # Set transformer network hyperparameters.
        default_transformer_kwargs = {
            "network": self.network,
            "euclidean_layer_idx": -2,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "num_classes": 10,
            "compile_kwargs": {},
            "fit_kwargs": {
                "epochs": self.epochs,
                # "callbacks": [EarlyStopping(patience=5, monitor="val_loss")],
                "verbose": self.verbose,
                "validation_split": 0.33,
                "batch_size": self.batch_size
            },
        }

        # Hyperparameter for KNN voter.
        default_voter_kwargs = {"k": int(np.log2(num_points_per_task))}

        self.pl = ProgressiveLearner(
            default_transformer_class=NeuralClassificationTransformer,
            default_transformer_kwargs=default_transformer_kwargs,
            default_voter_class=KNNClassificationVoter,
            default_voter_kwargs=default_voter_kwargs,
            default_decider_class=SimpleAverage,
            default_decider_kwargs={},
        )

    def add_task(self, X, y, task_id=None, decider_kwargs={}):

        if self.is_first_task:
            num_points_per_task = len(X)
            self.setup(num_points_per_task)
            self.is_first_task = False

        self.pl.add_task(
            X,
            y,
            task_id=task_id,
            transformer_voter_decider_split=[0.67, 0.33, 0.0],
            decider_kwargs=decider_kwargs,
        )

        return self

    def predict(self, X, task_id):
        return self.pl.predict(X, task_id)

    def predict_proba(self, X, task_id):
        return self.pl.predict_proba(X, task_id)
