"""
Main Author: Will LeVine
Corresponding Email: levinewill@icloud.com
"""
from .progressive_learner import ClassificationProgressiveLearner
from .transformers import TreeClassificationTransformer
from .voters import TreeClassificationVoter
from .deciders import SimpleArgmaxAverage

import numpy as np

from sklearn.ensemble import RandomForestClassifier
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

    n_jobs : int, default=1
        The number of jobs to run in parallel. ``-1`` means use all
        processors.

    max_samples : int or float, default=0.5
        The number of samples to draw from X (without replacement) to train
        each tree.
        - If None, then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1)`.

        Note: The number of samples used to learn the tree will be further
        reduced per the `tree_construction_proportion` value.

    honest_prior : {"ignore", "uniform", "empirical"}, default="ignore"
        Method for dealing with empty leaves during evaluation of a test
        sample. If "ignore", trees in which the leaf is empty are not used in
        the prediction. If "uniform", the prior tree posterior is 1/(number of
        classes). If "empirical", the prior tree posterior is the relative
        class frequency in the voting subsample. If all tree leaves are empty,
        "ignore" will use the empirical prior and the others will use their
        respective priors.

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
        max_samples=None,
        n_jobs=None,
        honest_prior="ignore",
    ):
        self.default_n_estimators = default_n_estimators
        self.default_tree_construction_proportion = default_tree_construction_proportion
        self.default_kappa = default_kappa
        self.default_max_depth = default_max_depth
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.honest_prior = honest_prior

        self.pl_ = ClassificationProgressiveLearner(
            default_transformer_class=TreeClassificationTransformer,
            default_transformer_kwargs={},
            default_voter_class=TreeClassificationVoter,
            default_voter_kwargs={"kappa": default_kappa},
            default_decider_class=SimpleArgmaxAverage,
            default_decider_kwargs={},
            n_jobs=n_jobs,
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
        transformer_kwargs={},
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

        TODO prune max_depth into transformer_kwargs
        max_depth : int or str, default='default'
            The maximum depth of a tree in the Lifelong Classification Forest.
            The default is used if 'default' is provided.

        transformer_kwargs : dict, default={}
            Additional named arguments to be passed to the transformer.

        Returns
        -------
        self : LifelongClassificationForest
            The object itself.
        """
        # TODO get rid of defaults in favor of None
        if n_estimators == "default":
            n_estimators = self.default_n_estimators
        if tree_construction_proportion == "default":
            tree_construction_proportion = self.default_tree_construction_proportion
        if kappa == "default":
            kappa = self.default_kappa
        if max_depth == "default":
            max_depth = self.default_max_depth

        # TODO eliminate by subsuming max_depth
        if not "fit_kwargs" in transformer_kwargs.keys():
            transformer_kwargs["fit_kwargs"] = {}
        transformer_kwargs["fit_kwargs"]["max_depth"] = max_depth

        X, y = check_X_y(X, y)
        if isinstance(self.max_samples, int):
            assert self.max_samples > 1
            max_samples = min(1, self.max_samples / X.shape[0])
        elif self.max_samples is None:
            max_samples = 1.0
        else:
            max_samples = self.max_samples
        return self.pl_.add_task(
            X,
            y,
            task_id=task_id,
            transformer_voter_decider_split=[
                tree_construction_proportion * max_samples,
                (1 - tree_construction_proportion) * max_samples,
                0,
            ],
            num_transformers=n_estimators,
            transformer_kwargs=transformer_kwargs,
            voter_kwargs={
                "classes" : np.unique(y),
                "kappa" : kappa,
                "honest_prior" : self.honest_prior
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
        X = check_array(X)
        return self.pl_.predict_proba(X, task_id)

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
    A class used to represent an Uncertainty Forest.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the UncertaintyForest

    kappa : float, default=np.inf
        The coefficient for finite sample correction. If set to the default
        value, finite sample correction is not performed.

    max_depth : int, default=None
        The maximum depth of a tree in the UncertaintyForest.

    tree_construction_proportion : float, default=0.5
        The proportions of the input data set aside to train each decision
        tree. The remainder of the data is used to fill in voting posteriors.

    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    poisson_sampler : boolean, default=False
        To match the GRF theory [#1grf]_, if True, the number of features
        considered at each tree are drawn from a poisson distribution with
        mean equal to `max_features`.

    n_jobs : int, default=None
        The number of jobs to run in parallel. ``-1`` means use all
        processors. None equates to 1.

    max_samples : int or float, default=None
        The number of samples to draw from X (without replacement) to train
        each tree.
        - If None, then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1)`.

        Note: The number of samples used to learn the tree will be further
        reduced per the `tree_construction_proportion` value.

    honest_prior : {"ignore", "uniform", "empirical"}, default="ignore"
        Method for dealing with empty leaves during evaluation of a test
        sample. If "ignore", trees in which the leaf is empty are not used in
        the prediction. If "uniform", the prior tree posterior is 1/(number of
        classes). If "empirical", the prior tree posterior is the relative
        class frequency in the voting subsample. If all tree leaves are empty,
        "ignore" will use the empirical prior and the others will use their
        respective priors.

    tree_kwargs : dict, default={}
        Named arguments to be passed to each
        sklearn.tree.DecisionTreeClassifier tree used in the construction
        of the forest in addition to the above parameters.

    Attributes
    ----------
    estimators_ : list of sklearn.tree.DecisionTreeClassifier
        The collection of fitted trees.

    voters_ : list of proglearn.voter.TreeClassificationVoter
        The collection of honest voters for leaves in matching trees in
        `self.estimators_` at the same index.

    lf_ : LifelongClassificationForest
        Internal LifelongClassificationForest used to train and make
        inference.

    n_features_ : int
        The number of features when `fit` is performed.

    tree_kwargs_ : dict
        Full set of keyword arguments passed to the Forest transformer.

    References
    ----------
    .. [#1grf] Athey, Susan, Julie Tibshirani and Stefan Wager.
    "Generalized Random Forests", Annals of Statistics, 2019.
    """

    def __init__(
        self,
        n_estimators=100,
        kappa=np.inf,
        max_depth=None,
        tree_construction_proportion=0.63,
        max_features="auto",
        poisson_sampler=False,
        max_samples=None,
        n_jobs=None,
        honest_prior="ignore",
        tree_kwargs={},
    ):
        self.n_estimators = n_estimators
        self.kappa = kappa
        self.max_depth = max_depth
        self.tree_construction_proportion = tree_construction_proportion
        self.max_features = max_features
        self.poisson_sampler = poisson_sampler
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.tree_kwargs = tree_kwargs
        self.honest_prior = honest_prior

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
        X, y = check_X_y(X, y)
        self.n_features_ = X.shape[1]
        self.lf_ = LifelongClassificationForest(
            default_n_estimators=self.n_estimators,
            default_kappa=self.kappa,
            default_max_depth=self.max_depth,
            default_tree_construction_proportion=self.tree_construction_proportion,
            max_samples=self.max_samples,
            n_jobs=self.n_jobs,
            honest_prior=self.honest_prior,
        )

        self.tree_kwargs_ = {
            "fit_kwargs": self.tree_kwargs,
            "max_features": self.max_features,
            "poisson_sampler": self.poisson_sampler,
        }
        self.lf_.add_task(
            X,
            y,
            task_id=0,
            transformer_kwargs=self.tree_kwargs_,
        )

        return self

    @property
    def estimators_(self):
        if not hasattr(self, "lf_"):
            raise AttributeError("Model has not been fitted. Please fit first.")
        return [t.transformer_ for t in self.lf_.pl_.transformer_id_to_transformers[0]]

    @property
    def voters_(self):
        if not hasattr(self, "lf_"):
            raise AttributeError("Model has not been fitted. Please fit first.")
        return [t for t in self.lf_.pl_.task_id_to_transformer_id_to_voters[0][0]]

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
        X = check_array(X)
        if not hasattr(self, "lf_"):
            raise AttributeError("Model has not been fitted. Please fit first.")
        return self.lf_.predict_proba(X, 0)

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
        X = check_array(X)
        if not hasattr(self, "lf_"):
            raise AttributeError("Model has not been fitted. Please fit first.")
        return self.lf_.predict(X, 0)
