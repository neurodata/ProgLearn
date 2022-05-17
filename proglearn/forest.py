"""
Main Author: Will LeVine
Corresponding Email: levinewill@icloud.com
"""
from .progressive_learner import ClassificationProgressiveLearner
from .transformers import TreeClassificationTransformer
from .voters import TreeClassificationVoter
from .deciders import SimpleArgmaxAverage

import numpy as np

from sklearn.utils.validation import check_X_y, check_array


class LifelongClassificationForest(ClassificationProgressiveLearner):
    """
    A class used to represent a lifelong classification forest.
    Parameters
    ----------
    default_n_estimators : int, default=100
        The number of trees used in the Lifelong Classification Forest
        used if 'n_estimators' is not fed to add_{task, transformer}.
    default_tree_construction_proportion : int, default=0.67
        The proportions of the input data set aside to train each decision
        tree. The remainder of the data is used to fill in voting posteriors.
        This is used if 'tree_construction_proportion' is not fed to add_task.
    default_kappa : float, default=np.inf
        The coefficient for finite sample correction.
        This is used if 'kappa' is not fed to add_task.
    default_max_depth : int, default=30
        The maximum depth of a tree in the Lifelong Classification Forest.
        This is used if 'max_depth' is not fed to add_task.
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
    default_transformer_class : TreeClassificationTransformer
        The class of transformer to which the forest defaults
        if None is provided in any of the functions which add or set
        transformers.
    default_transformer_kwargs : dict
        A dictionary with keys of type string and values of type obj corresponding
        to the given string kwarg. This determines to which type of transformer the
        forest defaults if None is provided in any of the functions
        which add or set transformers.
    default_voter_class : TreeClassificationVoter
        The class of voter to which the forest defaults
        if None is provided in any of the functions which add or set
        voters.
    default_voter_kwargs : dict
        A dictionary with keys of type string and values of type obj corresponding
        to the given string kwarg. This determines to which type of voter the
        forest defaults if None is provided in any of the functions
        which add or set voters.
    default_decider_class : SimpleArgmaxAverage
        The class of decider to which the forest defaults
        if None is provided in any of the functions which add or set
        deciders.
    default_decider_kwargs : dict
        A dictionary with keys of type string and values of type obj corresponding
        to the given string kwarg. This determines to which type of decider the
        forest defaults if None is provided in any of the functions
        which add or set deciders.
    """

    def __init__(
        self,
        default_n_estimators=100,
        default_tree_construction_proportion=0.67,
        default_kappa=np.inf,
        default_max_depth=30,
    ):
        super().__init__(
            default_transformer_class=TreeClassificationTransformer,
            default_transformer_kwargs={},
            default_voter_class=TreeClassificationVoter,
            default_voter_kwargs={"kappa": default_kappa},
            default_decider_class=SimpleArgmaxAverage,
            default_decider_kwargs={},
        )

        self.default_n_estimators = default_n_estimators
        self.default_tree_construction_proportion = default_tree_construction_proportion
        self.default_kappa = default_kappa
        self.default_max_depth = default_max_depth

    def add_task(
        self,
        X,
        y,
        task_id=None,
        n_estimators="default",
        tree_construction_proportion="default",
        kappa="default",
        max_depth="default",
        classes=None,
    ):
        """
        adds a task with id task_id, max tree depth max_depth, given input data matrix X
        and output data matrix y, to the Lifelong Classification Forest. Also splits
        data for training and voting based on tree_construction_proportion and uses the
        value of kappa to determine whether the learner will have
        finite sample correction.
        Parameters
        ----------
        X : ndarray
            The input data matrix.
        y : ndarray
            The output (response) data matrix.
        task_id : obj, default=None
            The id corresponding to the task being added.
        n_estimators : int or str, default='default'
            The number of trees used for the given task.
        tree_construction_proportion : int or str, default='default'
            The proportions of the input data set aside to train each decision
            tree. The remainder of the data is used to fill in voting posteriors.
            The default is used if 'default' is provided.
        kappa : float or str, default='default'
            The coefficient for finite sample correction.
            The default is used if 'default' is provided.
        max_depth : int or str, default='default'
            The maximum depth of a tree in the Lifelong Classification Forest.
            The default is used if 'default' is provided.
        Returns
        -------
        self : LifelongClassificationForest
            The object itself.
        """
        if n_estimators == "default":
            n_estimators = self.default_n_estimators
        if tree_construction_proportion == "default":
            tree_construction_proportion = self.default_tree_construction_proportion
        if kappa == "default":
            kappa = self.default_kappa
        if max_depth == "default":
            max_depth = self.default_max_depth

        X, y = check_X_y(X, y)
        return super().add_task(
            X,
            y,
            task_id=task_id,
            transformer_voter_decider_split=[
                tree_construction_proportion,
                1 - tree_construction_proportion,
                0,
            ],
            num_transformers=n_estimators,
            transformer_kwargs={"kwargs": {"max_depth": max_depth}},
            voter_kwargs={
                "classes": np.unique(y),
                "kappa": kappa,
            },
            decider_kwargs={"classes": np.unique(y)},
            classes=classes,
        )

    def add_transformer(
        self,
        X,
        y,
        transformer_id=None,
        n_estimators="default",
        max_depth="default",
        classes=None,
    ):
        """
        adds a transformer with id transformer_id and max tree depth max_depth, trained on
        given input data matrix, X, and output data matrix, y, to the Lifelong Classification Forest.
        Also trains the voters and deciders from new transformer to previous tasks, and will
        train voters and deciders from this transformer to all new tasks.
        Parameters
        ----------
        X : ndarray
            The input data matrix.
        y : ndarray
            The output (response) data matrix.
        transformer_id : obj, default=None
            The id corresponding to the transformer being added.
        n_estimators : int or str, default='default'
            The number of trees used for the given task.
        max_depth : int or str, default='default'
            The maximum depth of a tree in the Lifelong Classification Forest.
            The default is used if 'default' is provided.
        Returns
        -------
        self : LifelongClassificationForest
            The object itself.
        """
        if n_estimators == "default":
            n_estimators = self.default_n_estimators
        if max_depth == "default":
            max_depth = self.default_max_depth

        X, y = check_X_y(X, y)
        return super().add_transformer(
            X,
            y,
            transformer_kwargs={"kwargs": {"max_depth": max_depth}},
            transformer_id=transformer_id,
            num_transformers=n_estimators,
            classes=classes,
        )

    def update_task(
        self,
        X,
        y,
        task_id=None,
        n_estimators="default",
        tree_construction_proportion="default",
        kappa="default",
        max_depth="default",
        classes=None,
    ):
        """
        updates a task with id task_id, max tree depth max_depth, given input data matrix X
        and output data matrix y, to the Lifelong Classification Forest. Also splits
        data for training and voting based on tree_construction_proportion and uses the
        value of kappa to determine whether the learner will have
        finite sample correction.
        Parameters
        ----------
        X : ndarray
            The input data matrix.
        y : ndarray
            The output (response) data matrix.
        task_id : obj, default=None
            The id corresponding to the task being added.
        n_estimators : int or str, default='default'
            The number of trees used for the given task.
        tree_construction_proportion : int or str, default='default'
            The proportions of the input data set aside to train each decision
            tree. The remainder of the data is used to fill in voting posteriors.
            The default is used if 'default' is provided.
        kappa : float or str, default='default'
            The coefficient for finite sample correction.
            The default is used if 'default' is provided.
        max_depth : int or str, default='default'
            The maximum depth of a tree in the Lifelong Classification Forest.
            The default is used if 'default' is provided.
        Returns
        -------
        self : LifelongClassificationForest
            The object itself.
        """
        if n_estimators == "default":
            n_estimators = self.default_n_estimators
        if tree_construction_proportion == "default":
            tree_construction_proportion = self.default_tree_construction_proportion
        if kappa == "default":
            kappa = self.default_kappa
        if max_depth == "default":
            max_depth = self.default_max_depth

        X, y = check_X_y(X, y)

        return super().update_task(
            X,
            y,
            classes=classes,
            task_id=task_id,
            transformer_voter_decider_split=[
                tree_construction_proportion,
                1 - tree_construction_proportion,
                0,
            ],
            num_transformers=n_estimators,
            transformer_kwargs={"kwargs": {"max_depth": max_depth}},
            voter_kwargs={
                "classes": np.unique(y),
                "kappa": kappa,
            },
            decider_kwargs={"classes": np.unique(y)},
        )

    def update_transformer(
        self,
        X,
        y,
        classes=None,
        transformer_id=None,
        n_estimators="default",
        max_depth="default",
    ):

        if n_estimators == "default":
            n_estimators = self.default_n_estimators
        if max_depth == "default":
            max_depth = self.default_max_depth

        X, y = check_X_y(X, y)
        return super().update_transformer(
            X,
            y,
            classes=classes,
            transformer_kwargs={"kwargs": {"max_depth": max_depth}},
            transformer_id=transformer_id,
            num_transformers=n_estimators,
        )

    def predict_proba(self, X, task_id):
        """
        estimates class posteriors under task_id for each example in input data X.
        Parameters
        ----------
        X : ndarray
            The input data matrix.
        task_id:
            The id corresponding to the task being mapped to.
        Returns
        -------
        y_proba_hat : ndarray of shape [n_samples, n_classes]
            posteriors per example
        """
        return super().predict_proba(check_array(X), task_id)

    def predict(self, X, task_id):
        """
        predicts class labels under task_id for each example in input data X.
        Parameters
        ----------
        X : ndarray
            The input data matrix.
        task_id : obj
            The id corresponding to the task being mapped to.
        Returns
        -------
        y_hat : ndarray of shape [n_samples]
            predicted class label per example
        """
        return super().predict(check_array(X), task_id)


class UncertaintyForest(LifelongClassificationForest):
    """
    A class used to represent an uncertainty forest.
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the UncertaintyForest
    kappa : float, default=np.inf
        The coefficient for finite sample correction.
        If set to the default value, finite sample correction is not performed.
    max_depth : int, default=30
        The maximum depth of a tree in the UncertaintyForest
    tree_construction_proportion : float, default = 0.67
        The proportions of the input data set aside to train each decision
        tree. The remainder of the data is used to fill in voting posteriors.
    Attributes
    ----------
    default_transformer_class : TreeClassificationTransformer
        The class of transformer to which the forest defaults
        if None is provided in any of the functions which add or set
        transformers.
    default_transformer_kwargs : dict
        A dictionary with keys of type string and values of type obj corresponding
        to the given string kwarg. This determines to which type of transformer the
        forest defaults if None is provided in any of the functions
        which add or set transformers.
    default_voter_class : TreeClassificationVoter
        The class of voter to which the forest defaults
        if None is provided in any of the functions which add or set
        voters.
    default_voter_kwargs : dict
        A dictionary with keys of type string and values of type obj corresponding
        to the given string kwarg. This determines to which type of voter the
        forest defaults if None is provided in any of the functions
        which add or set voters.
    default_decider_class : SimpleArgmaxAverage
        The class of decider to which the forest defaults
        if None is provided in any of the functions which add or set
        deciders.
    default_decider_kwargs : dict
        A dictionary with keys of type string and values of type obj corresponding
        to the given string kwarg. This determines to which type of decider the
        forest defaults if None is provided in any of the functions
        which add or set deciders.
    """

    def __init__(
        self,
        n_estimators=100,
        kappa=np.inf,
        max_depth=30,
        tree_construction_proportion=0.67,
    ):
        super().__init__(
            default_n_estimators=n_estimators,
            default_tree_construction_proportion=tree_construction_proportion,
            default_kappa=kappa,
            default_max_depth=max_depth,
        )

    def fit(self, X, y):
        """
        fits forest to data X with labels y
        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The data that will be trained on
        y : array of shape [n_samples]
            The label for cluster membership of the given data
        Returns
        -------
        self : UncertaintyForest
            The object itself.
        """

        return super().add_task(X, y, task_id=0)

    def predict_proba(self, X):
        """
        estimates class posteriors for each example in input data X.
        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The data whose posteriors we are estimating.
        Returns
        -------
        y_proba_hat : ndarray of shape [n_samples, n_classes]
            posteriors per example
        """
        return super().predict_proba(X, 0)

    def predict(self, X):
        """
        predicts class labels for each example in input data X.
        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The data on which we are performing inference.
        Returns
        -------
        y_hat : ndarray of shape [n_samples]
            predicted class label per example
        """
        return super().predict(X, 0)
