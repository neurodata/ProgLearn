#%%
import numpy as np

from sklearn.utils.validation import NotFittedError

from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter

from numpy import testing


def generate_data(n=100):
    X = np.concatenate([np.zeros(n), np.ones(n)])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return X, y


class TestTreeClassificationVoter:
    def test_initialize(self):
        TreeClassificationVoter()
        assert True

    def test_predict_without_fit(self):
        X, y = generate_data()
        testing.assert_raises(
            NotFittedError, TreeClassificationVoter().predict_proba, X
        )

    def test_correct_predict(self):
        X, y = generate_data()

        voter = TreeClassificationVoter()
        voter.fit(X, y)

        y_hat = np.argmax(voter.predict_proba(X), axis=1)

        testing.assert_allclose(y, y_hat)


class TestKNNClassificationVoter:
    def test_initialize(self):
        KNNClassificationVoter()
        assert True

    def test_vote_without_fit(self):
        # generate random data
        X = np.random.randn(100, 3)
        testing.assert_raises(NotFittedError, KNNClassificationVoter().predict_proba, X)

    def test_correct_vote(self):
        # set random seed
        np.random.seed(0)

        # generate training data and classes
        X = np.concatenate((np.zeros(100), np.ones(100))).reshape(-1, 1)
        Y = np.concatenate((np.zeros(100), np.ones(100)))

        # train model
        kcv = KNNClassificationVoter(3)
        kcv.fit(X, Y)

        # generate testing data and class probability
        X_test = np.ones(6).reshape(-1, 1)
        Y_test = np.concatenate((np.zeros((6, 1)), np.ones((6, 1))), axis=1)

        # check if model predicts as expected
        testing.assert_allclose(Y_test, kcv.predict_proba(X_test), atol=1e-4)
