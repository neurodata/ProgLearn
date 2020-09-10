'''
Main Author: Will LeVine 
Corresponding Email: levinewill@icloud.com
'''
import numpy as np

from .progressive_learner import ProgressiveLearner
from .transformers import NeuralClassificationTransformer
from .voters import KNNClassificationVoter
from .deciders import SimpleAverage

from sklearn.utils import check_X_y, check_array
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

class LifelongClassificationNetwork:
    def __init__(
        self,
        network,
        loss="categorical_crossentropy",
        optimizer=Adam(3e-4),
        epochs=100,
        batch_size=32,
        verbose=False,
    ):
        self.network = network
        self.loss = loss
        self.epochs = epochs
        self.optimizer = optimizer
        self.verbose = verbose
        self.batch_size = batch_size
        
        # Set transformer network hyperparameters.
        default_transformer_kwargs = {
            "network": self.network,
            "euclidean_layer_idx": -2,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "fit_kwargs": {
                "epochs": self.epochs,
                "callbacks": [EarlyStopping(patience=5, monitor="val_loss")],
                "verbose": self.verbose,
                "validation_split": 0.33,
                "batch_size": self.batch_size
            },
        }

        self.pl = ProgressiveLearner(
            default_transformer_class=NeuralClassificationTransformer,
            default_transformer_kwargs=default_transformer_kwargs,
            default_voter_class=KNNClassificationVoter,
            default_voter_kwargs={},
            default_decider_class=SimpleAverage,
            default_decider_kwargs={},
        )

    def add_task(self, X, y, task_id=None, transformer_voter_decider_split=[0.67, 0.33, 0]):
        self.pl.add_task(
            X,
            y,
            task_id=task_id,
            transformer_voter_decider_split=transformer_voter_decider_split
        )

        return self

    def predict(self, X, task_id, transformer_ids=None):
        return self.pl.predict(X, task_id, transformer_ids=transformer_ids)

    def predict_proba(self, X, task_id, transformer_ids=None):
        return self.pl.predict_proba(X, task_id, transformer_ids=transformer_ids)
