"""
Main Author: Will LeVine
Corresponding Email: levinewill@icloud.com
"""
from .progressive_learner import ClassificationProgressiveLearner
from .transformers import (
    TreeClassificationTransformer,
    ObliqueTreeClassificationTransformer,
)
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

    oblique : bool, default=False
        Specifies if an oblique tree should used for the classifier or not.

    feature_combinations : float, default=2
        The feature combinations to use for the oblique split.
        Equal to the parameter 'L' in the sporf paper.

    max_features : float, default=1.0
        Controls the max number of features to consider for oblique split.
        The parameter 'd' in the sporf paper is equal to ceil(max_features * dimensions)

    Attributes
    ----------
    pl_ : ClassificationProgressiveLearner
        Internal ClassificationProgressiveLearner used to train and make
        inference.
    """

    def __init__(
        self,
        default_n_estimators=100,
        default_tree_construction_proportion=0.67,
        default_kappa=np.inf,
        default_max_depth=30,
        oblique=False,
        default_feature_combinations=2,
        default_max_features=1.0,
    ):
        self.default_n_estimators = default_n_estimators
        self.default_tree_construction_proportion = default_tree_construction_proportion
        self.default_kappa = default_kappa
        self.default_max_depth = default_max_depth
        self.oblique = oblique

        if oblique:
            default_transformer_class = ObliqueTreeClassificationTransformer
            self.default_feature_combinations = default_feature_combinations
            self.default_max_features = default_max_features

        else:
            default_transformer_class = TreeClassificationTransformer

        self.pl_ = ClassificationProgressiveLearner(
            default_transformer_class=default_transformer_class,
            default_transformer_kwargs={},
            default_voter_class=TreeClassificationVoter,
            default_voter_kwargs={"kappa": default_kappa},
            default_decider_class=SimpleArgmaxAverage,
            default_decider_kwargs={},
        )

    def add_task(
        self,
        X,
        y,
        task_id=None,
        n_estimators="default",
        tree_construction_proportion="default",
        kappa="default",
        max_depth="default",
        feature_combinations="default",
        max_features="default",
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

        feature_combinations : float, default=2
            The feature combinations to use for the oblique split.
            Equal to the parameter 'L' in the sporf paper.

        max_features : int, default=None
            The max number of features to consider for oblique split.
            Equal to the parameter 'd' in the sporf paper.

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

        if self.oblique:
            if feature_combinations == "default":
                feature_combinations = self.default_feature_combinations
            if max_features == "default":
                max_features = self.default_max_features

            transformer_kwargs = {
                "kwargs": {
                    "max_depth": max_depth,
                    "feature_combinations": feature_combinations,
                    "max_features": max_features,
                }
            }

        else:
            transformer_kwargs = {"kwargs": {"max_depth": max_depth}}

        X, y = check_X_y(X, y)
        return self.pl_.add_task(
            X,
            y,
            task_id=task_id,
            transformer_voter_decider_split=[
                tree_construction_proportion,
                1 - tree_construction_proportion,
                0,
            ],
            num_transformers=n_estimators,
            transformer_kwargs=transformer_kwargs,
            voter_kwargs={
                "classes": np.unique(y),
                "kappa": kappa,
            },
            decider_kwargs={"classes": np.unique(y)},
        )

    def add_transformer(
        self,
        X,
        y,
        transformer_id=None,
        n_estimators="default",
        max_depth="default",
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
        return self.pl_.add_transformer(
            X,
            y,
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
        return self.pl_.predict_proba(check_array(X), task_id)

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
        return self.pl_.predict(check_array(X), task_id)


class UncertaintyForest:
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
    lf_ : LifelongClassificationForest
        Internal LifelongClassificationForest used to train and make
        inference.
    """

    def __init__(
        self,
        n_estimators=100,
        kappa=np.inf,
        max_depth=30,
        tree_construction_proportion=0.67,
    ):
        self.n_estimators = n_estimators
        self.kappa = kappa
        self.max_depth = max_depth
        self.tree_construction_proportion = tree_construction_proportion

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
        self.lf_ = LifelongClassificationForest(
            default_n_estimators=self.n_estimators,
            default_kappa=self.kappa,
            default_max_depth=self.max_depth,
            default_tree_construction_proportion=self.tree_construction_proportion,
        )

        X, y = check_X_y(X, y)
        return self.lf_.add_task(X, y, task_id=0)

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
        return self.lf_.predict_proba(check_array(X), 0)

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
        return self.lf_.predict(check_array(X), 0)
