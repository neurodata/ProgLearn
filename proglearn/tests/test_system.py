#%%
import pytest
import numpy as np

from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import (
    TreeClassificationTransformer,
    NeuralClassificationTransformer,
)
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter


def generate_2d_rotation(theta=0, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)

    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    return R


def generate_gaussian_parity(
    n, mean=np.array([-1, -1]), cov_scale=1, angle_params=None, k=1, acorn=None
):
    if acorn is not None:
        np.random.seed(acorn)

    d = len(mean)

    if mean[0] == -1 and mean[1] == -1:
        mean = mean + 1 / 2 ** k

    mnt = np.random.multinomial(n, 1 / (4 ** k) * np.ones(4 ** k))
    cumsum = np.cumsum(mnt)
    cumsum = np.concatenate(([0], cumsum))

    Y = np.zeros(n)
    X = np.zeros((n, d))

    for i in range(2 ** k):
        for j in range(2 ** k):
            temp = np.random.multivariate_normal(
                mean, cov_scale * np.eye(d), size=mnt[i * (2 ** k) + j]
            )
            temp[:, 0] += i * (1 / 2 ** (k - 1))
            temp[:, 1] += j * (1 / 2 ** (k - 1))

            X[cumsum[i * (2 ** k) + j] : cumsum[i * (2 ** k) + j + 1]] = temp

            if i % 2 == j % 2:
                Y[cumsum[i * (2 ** k) + j] : cumsum[i * (2 ** k) + j + 1]] = 0
            else:
                Y[cumsum[i * (2 ** k) + j] : cumsum[i * (2 ** k) + j + 1]] = 1

    if d == 2:
        if angle_params is None:
            angle_params = np.random.uniform(0, 2 * np.pi)

        R = generate_2d_rotation(angle_params)
        X = X @ R

    else:
        raise ValueError("d=%i not implemented!" % (d))

    return X, Y.astype(int)


class TestSystem:
    def test_nxor(self):
        # tests proglearn on xor nxor simulation data
        np.random.seed(12345)

        reps = 10
        errors = np.zeros((4, reps), dtype=float)

        for ii in range(reps):
            default_transformer_class = TreeClassificationTransformer
            default_transformer_kwargs = {"kwargs": {"max_depth": 30}}

            default_voter_class = TreeClassificationVoter
            default_voter_kwargs = {}

            default_decider_class = SimpleArgmaxAverage
            default_decider_kwargs = {"classes": np.arange(2)}
            progressive_learner = ProgressiveLearner(
                default_transformer_class=default_transformer_class,
                default_transformer_kwargs=default_transformer_kwargs,
                default_voter_class=default_voter_class,
                default_voter_kwargs=default_voter_kwargs,
                default_decider_class=default_decider_class,
                default_decider_kwargs=default_decider_kwargs,
            )

            xor, label_xor = generate_gaussian_parity(
                750, cov_scale=0.1, angle_params=0
            )
            test_xor, test_label_xor = generate_gaussian_parity(
                1000, cov_scale=0.1, angle_params=0
            )

            nxor, label_nxor = generate_gaussian_parity(
                750, cov_scale=0.1, angle_params=np.pi / 2
            )
            test_nxor, test_label_nxor = generate_gaussian_parity(
                1000, cov_scale=0.1, angle_params=np.pi / 2
            )

            progressive_learner.add_task(xor, label_xor, num_transformers=10)
            progressive_learner.add_task(nxor, label_nxor, num_transformers=10)

            uf_task1 = progressive_learner.predict(
                test_xor, transformer_ids=[0], task_id=0
            )
            l2f_task1 = progressive_learner.predict(test_xor, task_id=0)
            uf_task2 = progressive_learner.predict(
                test_nxor, transformer_ids=[1], task_id=1
            )
            l2f_task2 = progressive_learner.predict(test_nxor, task_id=1)

            errors[0, ii] = 1 - np.mean(uf_task1 == test_label_xor)
            errors[1, ii] = 1 - np.mean(l2f_task1 == test_label_xor)
            errors[2, ii] = 1 - np.mean(uf_task2 == test_label_nxor)
            errors[3, ii] = 1 - np.mean(l2f_task2 == test_label_nxor)

        bte = np.mean(errors[0,]) / np.mean(
            errors[
                1,
            ]
        )
        fte = np.mean(errors[2,]) / np.mean(
            errors[
                3,
            ]
        )

        assert bte > 1 and fte > 1
