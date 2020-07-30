import numpy as np
from .base import ClassificationDecider


class ProgressiveLearner:
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

        self.transformer_id_to_voter_class = {}
        self.transformer_id_to_voter_kwargs = {}

        self.task_id_to_decider_class = {}
        self.task_id_to_decider_kwargs = {}

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

    def _get_split_idxs(self, ra, transformer_voter_decider_split):
        if transformer_voter_decider_split is None:
            return ra, ra
        else:
            split = [
                np.sum(np.array(transformer_voter_decider_split)[:2]),
                transformer_voter_decider_split[2],
            ]
            if np.sum(split) > 1:
                return [np.random.choice(ra, int(len(ra) * p)) for p in split]
            else:
                first_idx = np.random.choice(ra, int(len(ra) * split[0]))
                second_idx = np.random.choice(
                    np.delete(ra, first_idx), int(len(ra) * split[1])
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
        default_voter_class=None,
        default_voter_kwargs=None,
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

        self.transformer_id_to_voter_class[transformer_id] = default_voter_class
        self.transformer_id_to_voter_kwargs[transformer_id] = default_voter_kwargs

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
                transformer_id in list(self.transformer_id_to_voter_class.keys())
                and self.transformer_id_to_voter_class[transformer_id] is not None
            ):
                voter_class = self.transformer_id_to_voter_class[transformer_id]
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
                transformer_id in list(self.transformer_id_to_voter_kwargs.keys())
                and self.transformer_id_to_voter_kwargs[transformer_id] is not None
            ):
                voter_kwargs = self.transformer_id_to_voter_kwargs[transformer_id]
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

    def set_decider(
        self, task_id, transformer_ids, decider_class=None, decider_kwargs=None
    ):

        if decider_class is None:
            if self.default_decider_class is None:
                if task_id not in list(self.task_id_to_decider_class.keys()):
                    raise ValueError(
                        "decider_class is None and 'default_decider_class' is None."
                    )
                else:
                    decider_class = self.task_id_to_decider_class[task_id]
            else:
                decider_class = self.default_decider_class

        if decider_kwargs is None:
            if self.default_decider_kwargs is None:
                if task_id not in list(self.task_id_to_decider_kwargs.keys()):
                    raise ValueError(
                        "decider_kwargs is None and 'default_decider_kwargs' is None."
                    )
                else:
                    decider_kwargs = self.task_id_to_decider_kwargs[task_id]
            else:
                decider_kwargs = self.default_decider_kwargs

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
        self.task_id_to_decider[task_id].fit(
            X, y, transformer_id_to_transformers, transformer_id_to_voters
        )

        self.task_id_to_decider_class[task_id] = decider_class
        self.task_id_to_decider_kwargs[task_id] = decider_kwargs

    def add_transformer(
        self,
        X,
        y,
        transformer_data_proportion=1,
        transformer_voter_data_idx=None,
        transformer_id=None,
        num_transformers=1,
        transformer_class=None,
        transformer_kwargs=None,
        voter_class=None,
        voter_kwargs=None,
        backward_task_ids=None,
    ):

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
                    transformer_voter_data_idx, int(transformer_data_proportion * n)
                )
            else:
                transformer_data_idx = None
            self.set_transformer(
                transformer_id=transformer_id,
                transformer_data_idx=transformer_data_idx,
                transformer_class=transformer_class,
                transformer_kwargs=transformer_kwargs,
                default_voter_class=voter_class,
                default_voter_kwargs=voter_kwargs,
            )
            voter_data_idx = np.delete(range(len(X)), transformer_data_idx)
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

        if task_id is None:
            task_id = max(
                len(self.get_transformer_ids()), len(self.get_task_ids())
            )  # come up with something that has fewer collisions

        self.task_id_to_X[task_id] = X
        self.task_id_to_y[task_id] = y

        # split into transformer/voter and decider data
        transformer_voter_data_idx, decider_idx = self._get_split_idxs(
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
                voter_class=voter_class,
                voter_kwargs=voter_kwargs,
                backward_task_ids=backward_task_ids,
            )

        # train voters and decider from previous (and current) transformers to new task
        for transformer_id in (
            forward_transformer_ids
            if forward_transformer_ids
            else self.get_transformer_ids()
        ):
            self.set_voter(transformer_id=transformer_id, task_id=task_id)

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

    def predict(self, X, task_id, transformer_ids=None):
        return self.task_id_to_decider[task_id].predict(
            X, transformer_ids=transformer_ids
        )

    def predict_proba(self, X, task_id, transformer_ids=None):
        decider = self.task_id_to_decider[task_id]
        if issubclass(decider, ClassificationDecider):
            return self.task_id_to_decider[task_id].predict_proba(
                X, transformer_ids=transformer_ids
            )
        else:
            raise AttributeError(
                "Cannot call `predict_proba` on non-classification decider."
            )

