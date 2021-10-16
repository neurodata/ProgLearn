"""
Main Author: Will LeVine
Corresponding Email: levinewill@icloud.com
"""
import numpy as np

from .progressive_learner import ClassificationProgressiveLearner
from .transformers import NeuralClassificationTransformer
from .voters import KNNClassificationVoter
from .deciders import SimpleArgmaxAverage

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.utils.validation import check_X_y, check_array


class LifelongClassificationNetwork(ClassificationProgressiveLearner):
    """
    A class for progressive learning using Lifelong Learning Networks in a classification setting.

    Parameters
    ----------
    network: Keras model
        Transformer network used to map input to output.

    loss: string
        String name of the function used to calculate the loss between labels and predictions.

    optimizer: str or instance of keras.optimizers
        Algorithm used as the optimizer.

    epochs: int
        Number of times the entire training set is iterated over.

    batch_size: int
        Batch size used in the training of the network.

    verbose: bool
        Boolean indicating the production of detailed logging information during training of the
        network.

    default_network_construction_proportion: float, default = 0.67
        The proportions of the input data set aside to train each network. The remainder of the
        data is used to fill in voting posteriors. This is used if 'tree_construction_proportion'
        is not fed to add_task.

    Attributes
    ----------
    task_id_to_X : dict
        A dictionary with keys of type obj corresponding to task ids
        and values of type ndarray corresponding to the input data matrix X.
        This dictionary thus maps input data matrix to the task where posteriors
        are to be estimated.

    task_id_to_y : dict
        A dictionary with keys of type obj corresponding to task ids
        and values of type ndarray corresponding to output data matrix y.
        This dictionary thus maps output data matrix to the task where posteriors
        are to be estimated.

    transformer_id_to_X : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type ndarray corresponding to the output data matrix X.
        This dictionary thus maps input data matrix to a particular transformer.

    transformer_id_to_y : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type ndarray corresponding to the output data matrix y.
        This dictionary thus maps output data matrix to a particular transformer.

    transformer_id_to_transformers : dict
        A dictionary with keys of type obj corresponding to transformer ids
        and values of type obj corresponding to a transformer. This dictionary thus
        maps transformer ids to the corresponding transformers.

    task_id_to_trasnformer_id_to_voters : dict
        A nested dictionary with outer key of type obj, corresponding to task ids
        inner key of type obj, corresponding to transformer ids,
        and values of type obj, corresponding to a voter. This dictionary thus maps
        voters to a corresponding transformer assigned to a particular task.

    task_id_to_decider : dict
        A dictionary with keys of type obj, corresponding to task ids,
        and values of type obj corresponding to a decider. This dictionary thus
        maps deciders to a particular task.

    task_id_to_decider_class : dict
        A dictionary with keys of type obj corresponding to task ids
        and values of type obj corresponding to a decider class. This dictionary
        thus maps decider classes to a particular task id.

    task_id_to_voter_class : dict
        A dictionary with keys of type obj corresponding to task ids
        and values of type obj corresponding to a voter class. This dictionary thus
        maps voter classes to a particular task id.

    task_id_to_voter_kwargs : dict
        A dictionary with keys of type obj corresponding to task ids
        and values of type obj corresponding to a voter kwargs. This dictionary thus
        maps voter kwargs to a particular task id.

    task_id_to_decider_kwargs : dict
        A dictionary with keys of type obj corresponding to task ids
        and values of type obj corresponding to a decider kwargs. This dictionary
        thus maps decider kwargs to a particular task id.

    task_id_to_bag_id_to_voter_data_idx : dict
        A nested dictionary with outer keys of type obj corresponding to task ids
        inner keys of type obj corresponding to bag ids
        and values of type obj corresponding to voter data indices.
        This dictionary thus maps voter data indices to particular bags
        for particular tasks.

    task_id_to_decider_idx : dict
        A dictionary with keys of type obj corresponding to task ids
        and values of type obj corresponding to decider indices. This dictionary
        thus maps decider indices to particular tasks.

    default_transformer_class : NeuralClassificationTransformer
        The class of transformer to which the network defaults
        if None is provided in any of the functions which add or set
        transformers.

    default_transformer_kwargs : dict
        A dictionary with keys of type string and values of type obj corresponding
        to the given string kwarg. This determines to which type of transformer the
        network defaults if None is provided in any of the functions
        which add or set transformers.

    default_voter_class : KNNClassificationVoter
        The class of voter to which the network defaults
        if None is provided in any of the functions which add or set
        voters.

    default_voter_kwargs : dict
        A dictionary with keys of type string and values of type obj corresponding
        to the given string kwarg. This determines to which type of voter the
        network defaults if None is provided in any of the functions
        which add or set voters.

    default_decider_class : SimpleArgmaxAverage
        The class of decider to which the network defaults
        if None is provided in any of the functions which add or set
        deciders.

    default_decider_kwargs : dict
        A dictionary with keys of type string and values of type obj corresponding
        to the given string kwarg. This determines to which type of decider the
        network defaults if None is provided in any of the functions
        which add or set deciders.
    """

    def __init__(
        self,
        network,
        loss="categorical_crossentropy",
        optimizer=Adam(3e-4),
        epochs=100,
        batch_size=32,
        verbose=False,
        default_network_construction_proportion=0.67,
    ):
        self.network = network
        self.loss = loss
        self.epochs = epochs
        self.optimizer = optimizer
        self.verbose = verbose
        self.batch_size = batch_size
        self.default_network_construction_proportion = (
            default_network_construction_proportion
        )

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
                "batch_size": self.batch_size,
            },
        }

        super().__init__(
            default_transformer_class=NeuralClassificationTransformer,
            default_transformer_kwargs=default_transformer_kwargs,
            default_voter_class=KNNClassificationVoter,
            default_voter_kwargs={},
            default_decider_class=SimpleArgmaxAverage,
            default_decider_kwargs={},
        )

    def add_task(self, X, y, task_id=None, network_construction_proportion="default"):
        """
        adds a task with id task_id, given input data matrix X
        and output data matrix y, to the Lifelong Classification Network

        Parameters
        ----------
        X: ndarray
            Input data matrix.

        y: ndarray
            Output (response) data matrix.

        task_id: obj
            The id corresponding to the task being added.

        network_construction_proportion: float or str, default='default'
            The proportions of the input data set aside to train each network. The remainder of the
            data is used to fill in voting posteriors. The default is used if 'default' is provided.

        Returns
        -------
        self : LifelongClassificationNetwork
            The object itself.
        """
        if network_construction_proportion == "default":
            network_construction_proportion = self.network_construction_proportion

        X, y = check_X_y(X, y, ensure_2d=False)
        return super().add_task(
            X,
            y,
            task_id=task_id,
            transformer_voter_decider_split=[
                network_construction_proportion,
                1 - network_construction_proportion,
                0,
            ],
            decider_kwargs={"classes": np.unique(y)},
            voter_kwargs={"classes": np.unique(y)},
        )

    def add_transformer(self, X, y, transformer_id=None):
        """
        adds a transformer with id transformer_id, trained on given input data matrix, X
        and output data matrix, y, to the Lifelong Classification Network. Also
        trains the voters and deciders from new transformer to previous tasks, and will
        train voters and deciders from this transformer to all new tasks.

        Parameters
        ----------
        X: ndarray
            Input data matrix.

        y: ndarray
            Output (response) data matrix.

        transformer_id: obj
            The id corresponding to the transformer being added.

        Returns
        -------
        self : LifelongClassificationNetwork
            The object itself.
        """
        X, y = check_X_y(X, y, ensure_2d=False)
        return super().add_transformer(X, y, transformer_id=transformer_id)

    def predict(self, X, task_id):
        """
        Predicts class labels under task_id for each example in input data X.

        Parameters
        ----------
        X: ndarray
            Input data matrix.

        task_id: obj
            The task on which you are interested in performing inference.

        Returns
        -------
        y_hat : ndarray of shape [n_samples]
            predicted class label per example
        """
        return super().predict(check_array(X, ensure_2d=False), task_id)

    def predict_proba(self, X, task_id):
        """
        Estimates class posteriors under task_id for each example in input data X.

        Parameters
        ----------
        X: ndarray
            Input data matrix.

        task_id: obj
            The task on which you are interested in estimating posteriors.

        Returns
        -------
        y_proba_hat : ndarray of shape [n_samples, n_classes]
            posteriors per example
        """
        return super().predict_proba(check_array(X, ensure_2d=False), task_id)
