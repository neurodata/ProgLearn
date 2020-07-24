from .progressive_learner import ProgressiveLearner
from .transformers import NeuralRegressionTransformer
from .voters import NeuralRegressionVoter
from .deciders import LinearRegressionDecider, KNNRegressionDecider

from sklearn.utils import check_X_y, check_array
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


class LifelongRegressionNetwork():
    
    def __init__(
        self, 
        network, 
        decider='linear', 
        loss='mse', 
        epochs=100, 
        lr=1e-3,
        verbose=False
    ):
        self.network = network
        self.decider = decider
        self.loss = loss
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.is_first_task = True

    def setup(self):

        # Set transformer network hyperparameters.
        default_transformer_kwargs = {
            "network" : self.network, 
            "euclidean_layer_idx" : -2,
            "loss" : self.loss,
            "optimizer" : Adam(self.lr),
            "compile_kwargs" : {},
            "fit_kwargs" : {
                "epochs" : self.epochs, 
                "verbose" : self.verbose,
                "callbacks": [EarlyStopping(patience=10, monitor="val_loss")],
                "validation_split": 0.25,
            }
        }

        # Set voter network hyperparameters.
        default_voter_kwargs = {
            "validation_split" : 0.25,
            "loss" : self.loss,
            "lr" : self.lr,
            "epochs" : self.epochs,
            "verbose" : self.verbose
        }

        # Choose decider.
        if self.decider == 'linear':
            default_decider_class = LinearRegressionDecider
        elif self.decider == 'knn':
            default_decider_class = KNNRegressionDecider
        else:
            raise ValueError("Decider must be 'linear' or 'knn'.")

        self.pl = ProgressiveLearner(
            default_transformer_class=NeuralRegressionTransformer, 
            default_transformer_kwargs=default_transformer_kwargs,
            default_voter_class=NeuralRegressionVoter,
            default_voter_kwargs=default_voter_kwargs, 
            default_decider_class=default_decider_class,
            default_decider_kwargs={}
        )
        
    def add_task(self, X, y, task_id=None):

        if self.is_first_task:
            self.setup()
            self.is_first_task = False

        X, y = check_X_y(X, y)
        self.pl.add_task(
            X, 
            y, 
            task_id=task_id, 
            transformer_voter_decider_split=[0.6, 0.3, 0.1]
        )

        return self
        
    def predict(self, X, task_id):
        X = check_array(X)
        return self.pl.predict(X, task_id)
