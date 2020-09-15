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
    """
    A class for progressive learning using Lifelong Learning Networks in a classification setting. 
    
    Parameters
    ----------
    network: NeuralClassificationTransformer object
        Transformer network used to map input to output. 
    loss: string
        String name of the function used to calculate the loss between labels and predictions. 
    optimizer: Optimizer object
        Algorithm used as the optimizer.
    epochs: int
        Number of times the entire training set is iterated over. 
    batch_size: int
        Number of samples iterated through before making a prediction and generating error. 
    verbose: bool
        Boolean indicating the production of detailed logging information. 
    """
    
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
        )

    def add_task(self, X, y, task_id=None, transformer_voter_decider_split=[0.67, 0.33, 0]):
        """
        Add a new task to the progressive learner. 
        
        Parameters
        ----------
        X: ndarray
            Input data matrix.
        y: ndarray
            Output (response) data matrix. 
        task_id: obj
            The id corresponding to the task being added. 
        transformer_voter_decider_split: ndarray
            Array corresponding to the proportions of data used to train the transformer(s) corresponding to 
            the task_id, to train the voter(s) from the transformer(s) to the task_id, and to train the decider, respectively.
        """
        self.pl.add_task(
            X,
            y,
            task_id=task_id,
            transformer_voter_decider_split=transformer_voter_decider_split,
            decider_kwargs = {"classes" : np.unique(y)}
        )

        return self
    
    def add_transformer(self, X, y, transformer_id=None):
        """ 
        Add a new transformer corresponding to the transformer_id. 
        
        Parameters
        ----------
        X: ndarray
            Input data matrix.
        y: ndarray
            Output (response) data matrix. 
        transformer_id: obj
            The transformer you are interested in adding to the progressive learner. 
        """
        
        self.pl.add_transformer(
            X,
            y,
            transformer_id=transformer_id
        )

        return self

    def predict(self, X, task_id):
        """
        Perform inference corresponding to the input task_id using the progressive learner. 
        
        Parameters
        ----------
        X: ndarray
            Input data matrix.
        task_id: obj
            The task on which you are interested in performing inference. 
        """
        
        return self.pl.predict(X, task_id)

    def predict_proba(self, X, task_id):
        """
        Estimate posteriors under a given task_id using the decider. 
        
        Parameters
        ----------
        X: ndarray
            Input data matrix. 
        task_id: obj
            The task on which you are interested in estimating posteriors. 
        """
        
        return self.pl.predict_proba(X, task_id)
