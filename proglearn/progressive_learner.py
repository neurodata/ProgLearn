"""
Main Author: Will LeVine 
Corresponding Email: levinewill@icloud.com
"""
import numpy as np
from .base import BaseClassificationProgressiveLearner, BaseProgressiveLearner


class ProgressiveLearner(BaseProgressiveLearner):
    """
    A (mostly) internal class for progressive learning. Most users who desire to
    utilize ProgLearn should use the classes defined in {network, forest}.py instead
    of this class.

    Parameters
    ----------
    default_transformer_class : BaseTransformer, default=None
        The class of transformer to which the progressive learner defaults
        if None is provided in any of the functions which add or set
        transformers.
    default_transformer_kwargs : dict, default=None
        A dictionary with keys of type string and values of type obj corresponding
        to the given string kwarg. This determines to which type of transformer the
        progressive learner defaults if None is provided in any of the functions
        which add or set transformers.
    default_voter_class : BaseVoter, default=None
        The class of voter to which the progressive learner defaults
        if None is provided in any of the functions which add or set
        voters.
    default_voter_kwargs : dict, default=None
        A dictionary with keys of type string and values of type obj corresponding
        to the given string kwarg. This determines to which type of voter the
        progressive learner defaults if None is provided in any of the functions
        which add or set voters.
    default_decider_class : BaseDecider, default=None
        The class of decider to which the progressive learner defaults
        if None is provided in any of the functions which add or set
        deciders.
    default_decider_kwargs : dict, default=None
        A dictionary with keys of type string and values of type obj corresponding
        to the given string kwarg. This determines to which type of decider the
        progressive learner defaults if None is provided in any of the functions
        which add or set deciders.

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

    default_transformer_class : BaseTransformer
        Stores the default transformer class as specified by the parameter
        default_transformer_class.

    default_transformer_kwargs : dict
        Stores the default transformer kwargs as specified by the parameter
        default_transformer_kwargs.

    default_voter_class : BaseVoter
        Stores the default voter class as specified by the parameter
        default_voter_class.

    default_voter_kwargs : dict
        Stores the default voter kwargs as specified by the parameter
        default_voter_kwargs.

    default_decider_class : BaseDecider
        Stores the default decider class as specified by the parameter
        default_decider_class.

    default_decider_kwargs : dict
        Stores the default decider kwargs as specified by the parameter
        default_decider_kwargs.
    """

    def __init__(
        self,
        default_transformer_class=None,
        default_transformer_kwargs=None,
        default_voter_class=None,
        default_voter_kwargs=None,
        default_decider_class=None,
        default_decider_kwargs=None,
    ):

        (
            self.task_id_to_X,
            self.task_id_to_y,
            self.transformer_id_to_X,
            self.transformer_id_to_y,
        ) = ({}, {}, {}, {})

        self.transformer_id_to_transformers = {}
        self.task_id_to_transformer_id_to_voters = {}
        self.task_id_to_decider = {}

        self.task_id_to_decider_class = {}
        self.task_id_to_decider_kwargs = {}

        self.task_id_to_voter_class = {}
        self.task_id_to_voter_kwargs = {}

        self.task_id_to_bag_id_to_voter_data_idx = {}
        self.task_id_to_decider_idx = {}

        self.default_transformer_class = default_transformer_class
        self.default_transformer_kwargs = default_transformer_kwargs

        self.default_voter_class = default_voter_class
        self.default_voter_kwargs = default_voter_kwargs

        self.default_decider_class = default_decider_class
        self.default_decider_kwargs = default_decider_kwargs

    def get_transformer_ids(self):
        return np.array(list(self.transformer_id_to_transformers.keys()))

    def get_task_ids(self):
        return np.array(list(self.task_id_to_decider.keys()))

    def _append_transformer(self, transformer_id, transformer):
        if transformer_id in self.get_transformer_ids():
            self.transformer_id_to_transformers[transformer_id].append(transformer)
        else:
            self.transformer_id_to_transformers[transformer_id] = [transformer]

    def _append_voter(self, transformer_id, task_id, voter):
        if task_id in list(self.task_id_to_transformer_id_to_voters.keys()):
            if transformer_id in list(
                self.task_id_to_transformer_id_to_voters[task_id].keys()
            ):
                self.task_id_to_transformer_id_to_voters[task_id][
                    transformer_id
                ].append(voter)
            else:
                self.task_id_to_transformer_id_to_voters[task_id][transformer_id] = [
                    voter
                ]
        else:
            self.task_id_to_transformer_id_to_voters[task_id] = {
                transformer_id: [voter]
            }

    def _append_voter_data_idx(self, task_id, bag_id, voter_data_idx):
        if task_id in list(self.task_id_to_bag_id_to_voter_data_idx.keys()):
            self.task_id_to_bag_id_to_voter_data_idx[task_id][bag_id] = voter_data_idx
        else:
            self.task_id_to_bag_id_to_voter_data_idx[task_id] = {bag_id: voter_data_idx}

    def _append_decider_idx(self, task_id, decider_idx):
        self.task_id_to_decider_idx[task_id] = decider_idx

    def _bifurcate_decider_idxs(self, ra, transformer_voter_decider_split):
        if transformer_voter_decider_split is None:
            return ra, ra
        else:
            split = [
                np.sum(np.array(transformer_voter_decider_split)[:2]),
                transformer_voter_decider_split[2],
            ]
            if np.sum(split) > 1:
                return [
                    np.random.choice(ra, int(len(ra) * p), replace=False) for p in split
                ]
            else:
                first_idx = np.random.choice(ra, int(len(ra) * split[0]), replace=False)
                second_idx = np.random.choice(
                    np.delete(ra, first_idx), int(len(ra) * split[1]), replace=False
                )
                return first_idx, second_idx

    # make sure the below ganular functions work without add_{transformer, task}
    def set_transformer(
        self,
        transformer_id=None,
        transformer=None,
        transformer_data_idx=None,
        transformer_class=None,
        transformer_kwargs=None,
    ):

        if transformer_id is None:
            transformer_id = len(self.get_transformer_ids())

        X = (
            self.transformer_id_to_X[transformer_id]
            if transformer_id in list(self.transformer_id_to_X.keys())
            else self.task_id_to_X[transformer_id]
        )
        y = (
            self.transformer_id_to_y[transformer_id]
            if transformer_id in list(self.transformer_id_to_y.keys())
            else self.task_id_to_y[transformer_id]
        )
        if transformer_data_idx is not None:
            X, y = X[transformer_data_idx], y[transformer_data_idx]

        if X is None and y is None:
            if transformer.is_fitted():
                self._append_transformer(transformer_id, transformer)
            else:
                raise ValueError(
                    "transformer_class is not fitted and X is None and y is None."
                )
            return

        # Type check X

        if transformer_class is None:
            if self.default_transformer_class is None:
                raise ValueError(
                    "transformer_class is None and 'default_transformer_class' is None."
                )
            else:
                transformer_class = self.default_transformer_class

        if transformer_kwargs is None:
            if self.default_transformer_kwargs is None:
                raise ValueError(
                    """transformer_kwargs is None and 
                    'default_transformer_kwargs' is None."""
                )
            else:
                transformer_kwargs = self.default_transformer_kwargs

        # Fit transformer and new voter
        if y is None:
            self._append_transformer(
                transformer_id, transformer_class(**transformer_kwargs).fit(X)
            )
        else:
            # Type check y
            self._append_transformer(
                transformer_id, transformer_class(**transformer_kwargs).fit(X, y)
            )

    def set_voter(
        self,
        transformer_id,
        task_id=None,
        voter_class=None,
        voter_kwargs=None,
        bag_id=None,
    ):

        # Type check X

        # Type check y

        if task_id is None:
            task_id = len(self.get_task_ids())

        if voter_class is None:
            if (
                task_id in list(self.task_id_to_voter_class.keys())
                and self.task_id_to_voter_class[task_id] is not None
            ):
                voter_class = self.task_id_to_voter_class[task_id]
            elif self.default_voter_class is not None:
                voter_class = self.default_voter_class
            else:
                raise ValueError(
                    """voter_class is None, the default voter class for the overall 
                    learner is None, and the default voter class 
                    for this transformer is None."""
                )

        if voter_kwargs is None:
            if (
                task_id in list(self.task_id_to_voter_kwargs.keys())
                and self.task_id_to_voter_kwargs[task_id] is not None
            ):
                voter_kwargs = self.task_id_to_voter_kwargs[task_id]
            elif self.default_voter_kwargs is not None:
                voter_kwargs = self.default_voter_kwargs
            else:
                raise ValueError(
                    """voter_kwargs is None, the default voter kwargs for the overall 
                    learner is None, and the default voter kwargs 
                    for this transformer is None."""
                )

        X = self.task_id_to_X[task_id]
        y = self.task_id_to_y[task_id]
        if bag_id is None:
            transformers = self.transformer_id_to_transformers[transformer_id]
        else:
            transformers = [self.transformer_id_to_transformers[transformer_id][bag_id]]
        for transformer_num, transformer in enumerate(transformers):
            if transformer_id == task_id:
                voter_data_idx = self.task_id_to_bag_id_to_voter_data_idx[task_id][
                    transformer_num
                ]
            else:
                voter_data_idx = np.delete(
                    range(len(X)), self.task_id_to_decider_idx[task_id]
                )
            self._append_voter(
                transformer_id,
                task_id,
                voter_class(**voter_kwargs).fit(
                    transformer.transform(X[voter_data_idx]), y[voter_data_idx]
                ),
            )

        self.task_id_to_voter_class[task_id] = voter_class
        self.task_id_to_voter_kwargs[task_id] = voter_kwargs

    def set_decider(
        self, task_id, transformer_ids, decider_class=None, decider_kwargs=None
    ):
        if decider_class is None:
            if task_id in list(self.task_id_to_decider_class.keys()):
                decider_class = self.task_id_to_decider_class[task_id]
            elif self.default_decider_class is not None:
                decider_class = self.default_decider_class
            else:
                raise ValueError(
                    "decider_class is None and 'default_decider_class' is None."
                )
        if decider_kwargs is None:
            if task_id in list(self.task_id_to_decider_kwargs.keys()):
                decider_kwargs = self.task_id_to_decider_kwargs[task_id]
            elif self.default_decider_kwargs is not None:
                decider_kwargs = self.default_decider_kwargs
            else:
                raise ValueError(
                    "decider_kwargs is None and 'default_decider_kwargs' is None."
                )

        transformer_id_to_transformers = {
            transformer_id: self.transformer_id_to_transformers[transformer_id]
            for transformer_id in transformer_ids
        }
        transformer_id_to_voters = {
            transformer_id: self.task_id_to_transformer_id_to_voters[task_id][
                transformer_id
            ]
            for transformer_id in transformer_ids
        }

        X, y = self.task_id_to_X[task_id], self.task_id_to_y[task_id]

        self.task_id_to_decider[task_id] = decider_class(**decider_kwargs)
        decider_idx = self.task_id_to_decider_idx[task_id]
        self.task_id_to_decider[task_id].fit(
            X[decider_idx],
            y[decider_idx],
            transformer_id_to_transformers,
            transformer_id_to_voters,
        )

        self.task_id_to_decider_class[task_id] = decider_class
        self.task_id_to_decider_kwargs[task_id] = decider_kwargs

    def add_transformer(
        self,
        X,
        y,
        transformer_data_proportion=1.0,
        transformer_voter_data_idx=None,
        transformer_id=None,
        num_transformers=1,
        transformer_class=None,
        transformer_kwargs=None,
        backward_task_ids=None,
    ):
        """
        Adds a transformer to the progressive learner and trains the voters and
        deciders from this new transformer to the specified backward_task_ids.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        y : ndarray
            Output (response) data matrix.

        transformer_data_proportion : float, default=1.0
            The proportion of the data set aside to train the transformer. The
            remainder of the data is used to train voters. This is used in the
            case that you are using a bagging algorithm and want the various
            components in that bagging ensemble to train on disjoint subsets of
            the data. This parameter is mostly for internal use.

        transformer_voter_data_idx : ndarray, default=None
            A 1d array of type int used to specify the aggregate indices of the input
            data used to train the transformers and voters. This is used in the
            case that X and/or y contain data that you do not want to use to train
            transformers or voters (e.g. X and/or y contains decider training data
            disjoint from the transformer/voter data). This parameter is mostly
            for internal use.

        transformer_id : obj, default=None
            The id corresponding to the transformer being added.

        num_transformers : int, default=1
            The number of transformers to add corresponding to the given inputs.

        transformer_class : BaseTransformer, default=None
            The class of the transformer(s) being added.

        transformer_kwargs : dict, default=None
            A dictionary with keys of type string and values of type obj corresponding
            to the given string kwarg. This determines the kwargs of the transformer(s)
            being added.

        backward_task_ids : ndarray, default=None
            A 1d array of type obj used to specify to which existing task voters and deciders
            will be trained from the transformer(s) being added.

        Returns
        -------
        self : ProgressiveLearner
            The object itself.
        """
        if transformer_id is None:
            transformer_id = len(self.get_transformer_ids())

        backward_task_ids = (
            backward_task_ids if backward_task_ids is not None else self.get_task_ids()
        )
        transformer_voter_data_idx = (
            range(len(X))
            if transformer_voter_data_idx is None
            else transformer_voter_data_idx
        )

        if transformer_id not in list(self.task_id_to_X.keys()):
            self.transformer_id_to_X[transformer_id] = X
        if transformer_id not in list(self.task_id_to_y.keys()):
            self.transformer_id_to_y[transformer_id] = y

        # train new transformers
        for transformer_num in range(num_transformers):
            if X is not None:
                n = len(X)
            elif y is not None:
                n = len(y)
            else:
                n = None
            if n is not None:
                transformer_data_idx = np.random.choice(
                    transformer_voter_data_idx,
                    int(transformer_data_proportion * n),
                    replace=False,
                )
            else:
                transformer_data_idx = None
            self.set_transformer(
                transformer_id=transformer_id,
                transformer_data_idx=transformer_data_idx,
                transformer_class=transformer_class,
                transformer_kwargs=transformer_kwargs,
            )
            voter_data_idx = np.delete(transformer_voter_data_idx, transformer_data_idx)
            self._append_voter_data_idx(
                task_id=transformer_id,
                bag_id=transformer_num,
                voter_data_idx=voter_data_idx,
            )

        # train voters and deciders from new transformer to previous tasks
        for existing_task_id in np.intersect1d(backward_task_ids, self.get_task_ids()):
            self.set_voter(transformer_id=transformer_id, task_id=existing_task_id)
            self.set_decider(
                task_id=existing_task_id,
                transformer_ids=list(
                    self.task_id_to_transformer_id_to_voters[existing_task_id].keys()
                ),
            )

        return self

    def add_task(
        self,
        X,
        y,
        task_id=None,
        transformer_voter_decider_split=[0.67, 0.33, 0],
        num_transformers=1,
        transformer_class=None,
        transformer_kwargs=None,
        voter_class=None,
        voter_kwargs=None,
        decider_class=None,
        decider_kwargs=None,
        backward_task_ids=None,
        forward_transformer_ids=None,
    ):
        """
        Adds a task to the progressive learner. Optionally trains one or more
        transformer from the input data (if num_transformers > 0), adds voters
        and deciders from this/these new transformer(s) to the tasks specified
        in backward_task_ids, and adds voters and deciders from the transformers
        specified in forward_transformer_ids (and from the newly added transformer(s)
        corresponding to the input task_id if num_transformers > 0) to the
        new task_id.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        y : ndarray
            Output (response) data matrix.

        task_id : obj, default=None
            The id corresponding to the task being added.

        transformer_voter_decider_split : ndarray, default=[0.67, 0.33, 0]
            A 1d array of length 3. The 0th index indicates the proportions of the input
            data used to train the (optional) newly added transformer(s) corresponding to
            the task_id provided in this function call. The 1st index indicates the proportion of
            the data set aside to train the voter(s) from these (optional) newly added
            transformer(s) to the task_id provided in this function call. For all other tasks,
            the aggregate transformer and voter data pairs from those tasks are used to train
            the voter(s) from these (optional) newly added transformer(s) to those tasks;
            for all other transformers, the aggregate transformer and voter data provided in
            this function call is used to train the voter(s) from those transformers to
            the task_id provided in this function call. The 2nd index indicates the
            proportion of the data set aside to train the decider - these indices are saved
            internally and will be used to train all further deciders corresponding to this
            task for all function calls.

        num_transformers : int, default=1
            The number of transformers to add corresponding to the given inputs.

        transformer_class : BaseTransformer, default=None
            The class of the transformer(s) being added.

        transformer_kwargs : dict, default=None
            A dictionary with keys of type string and values of type obj corresponding
            to the given string kwarg. This determines the kwargs of the transformer(s)
            being added.

        voter_class : BaseVoter, default=None
            The class of the voter(s) being added.

        voter_kwargs : dict, default=None
            A dictionary with keys of type string and values of type obj corresponding
            to the given string kwarg. This determines the kwargs of the voter(s)
            being added.

        decider_class : BaseDecider, default=None
            The class of the decider(s) being added.

        decider_kwargs : dict, default=None
            A dictionary with keys of type string and values of type obj corresponding
            to the given string kwarg. This determines the kwargs of the decider(s)
            being added.

        backward_task_ids : ndarray, default=None
            A 1d array of type obj used to specify to which existing task voters and deciders
            will be trained from the transformer(s) being added.

        foward_transformer_ids : ndarray, default=None
            A 1d array of type obj used to specify from which existing transformer(s) voters and
            deciders will be trained to the new task. If num_transformers > 0, the input task_id
            corresponding to the task being added is automatically appended to this 1d array.

        Returns
        -------
        self : ProgressiveLearner
            The object itself.
        """
        if task_id is None:
            task_id = max(
                len(self.get_transformer_ids()), len(self.get_task_ids())
            )  # come up with something that has fewer collisions

        self.task_id_to_X[task_id] = X
        self.task_id_to_y[task_id] = y

        # split into transformer/voter and decider data
        transformer_voter_data_idx, decider_idx = self._bifurcate_decider_idxs(
            range(len(X)), transformer_voter_decider_split
        )
        self._append_decider_idx(task_id, decider_idx)

        # add new transformer and train voters and decider
        # from new transformer to previous tasks
        if num_transformers > 0:
            self.add_transformer(
                X,
                y,
                transformer_data_proportion=transformer_voter_decider_split[0]
                if transformer_voter_decider_split
                else 1,
                transformer_voter_data_idx=transformer_voter_data_idx,
                transformer_id=task_id,
                num_transformers=num_transformers,
                transformer_class=transformer_class,
                transformer_kwargs=transformer_kwargs,
                backward_task_ids=backward_task_ids,
            )

        # train voters and decider from previous (and current) transformers to new task
        for transformer_id in (
            forward_transformer_ids
            if forward_transformer_ids
            else self.get_transformer_ids()
        ):
            self.set_voter(
                transformer_id=transformer_id,
                task_id=task_id,
                voter_class=voter_class,
                voter_kwargs=voter_kwargs,
            )

        # train decider of new task
        if forward_transformer_ids:
            if num_transformers == 0:
                transformer_ids = forward_transformer_ids
            else:
                transformer_ids = np.concatenate([forward_transformer_ids, task_id])
        else:
            transformer_ids = self.get_transformer_ids()
        self.set_decider(
            task_id=task_id,
            transformer_ids=transformer_ids,
            decider_class=decider_class,
            decider_kwargs=decider_kwargs,
        )

        return self

    def predict(self, X, task_id, transformer_ids=None):
        """
        predicts labels under task_id for each example in input data X
        using the given transformer_ids.

        Parameters
        ----------
        X : ndarray
            The input data matrix.

        task_id : obj
            The id corresponding to the task being mapped to.

        transformer_ids : list, default=None
            The list of transformer_ids through which a user would like
            to send X (which will be pipelined with their corresponding
            voters) to make an inference prediction.

        Returns
        -------
        y_hat : ndarray of shape [n_samples]
            predicted class label per example
        """
        return self.task_id_to_decider[task_id].predict(
            X, transformer_ids=transformer_ids
        )


class ClassificationProgressiveLearner(
    BaseClassificationProgressiveLearner, ProgressiveLearner
):
    """
    A (mostly) internal class for progressive learning in the classification
    setting. Most users who desire to utilize ProgLearn should use the classes
    defined in {network, forest}.py instead of this class.
    """

    def predict_proba(self, X, task_id, transformer_ids=None):
        """
        predicts posteriors under task_id for each example in input data X
        using the given transformer_ids.

        Parameters
        ----------
        X : ndarray
            The input data matrix.

        task_id : obj
            The id corresponding to the task being mapped to.

        transformer_ids : list, default=None
            The list of transformer_ids through which a user would like
            to send X (which will be pipelined with their corresponding
            voters) to estimate posteriors.

        Returns
        -------
        y_proba_hat : ndarray of shape [n_samples, n_classes]
            posteriors per example
        """
        decider = self.task_id_to_decider[task_id]
        return self.task_id_to_decider[task_id].predict_proba(
            X, transformer_ids=transformer_ids
        )
