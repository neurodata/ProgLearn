import numpy as np
from scipy.spatial import distance
import sklearn.ensemble
from proglearn.sims import generate_gaussian_parity
import random
import math


def bootstrap(angle_sweep=range(0, 90, 5), n_samples=100, reps=1000):
    """
    Runs getPval many times to perform a bootstrap exeriment.
    """
    p_vals = []
    # generate xor
    X_xor, y_xor = generate_gaussian_parity(n_samples, angle_params=0)
    for angle in angle_sweep:
        # print('Processing angle:', angle)
        # we can use the same xor as from above but we need a new rxor
        # generate rxor with different angles

        X_rxor, y_rxor = generate_gaussian_parity(
            n_samples, angle_params=math.radians(angle)
        )

        # we want to pick 70 samples from xor/rxor to train trees so we need to first subset each into arrays with only xor_0/1 and rxor_0/1
        X_xor_0 = X_xor[np.where(y_xor == 0)]
        X_xor_1 = X_xor[np.where(y_xor == 1)]

        X_rxor_0 = X_rxor[np.where(y_rxor == 0)]
        X_rxor_1 = X_rxor[np.where(y_rxor == 1)]

        # we can concat the first 35 samples from each pair to use to tatal 70 samples for training and 30 for predict proba
        X_xor_train = np.concatenate((X_xor_0[0:35], X_xor_1[0:35]))
        y_xor_train = np.concatenate((np.zeros(35), np.ones(35)))

        # repeat for rxor
        X_rxor_train = np.concatenate((X_rxor_0[0:35], X_rxor_1[0:35]))
        y_rxor_train = np.concatenate((np.zeros(35), np.ones(35)))

        # make sure X_rxor_train is the right size everytime, run into errors sometime
        while len(X_rxor_train) != 70:
            X_rxor, y_rxor = generate_gaussian_parity(
                n_samples, angle_params=math.radians(angle)
            )
            # we want to pick 70 samples from xor/rxor to train trees so we need to first subset each into arrays with only xor_0/1 and rxor_0/1
            X_xor_0 = X_xor[np.where(y_xor == 0)]
            X_xor_1 = X_xor[np.where(y_xor == 1)]

            X_rxor_0 = X_rxor[np.where(y_rxor == 0)]
            X_rxor_1 = X_rxor[np.where(y_rxor == 1)]

            # we can concat the first 35 samples from each pair to use to tatal 70 samples for training and 30 for predict proba
            X_xor_train = np.concatenate((X_xor_0[0:35], X_xor_1[0:35]))
            y_xor_train = np.concatenate((np.zeros(35), np.ones(35)))

            # repeat for rxor
            X_rxor_train = np.concatenate((X_rxor_0[0:35], X_rxor_1[0:35]))
            y_rxor_train = np.concatenate((np.zeros(35), np.ones(35)))

        # init the rf's
        # xor rf
        clf_xor = sklearn.ensemble.RandomForestClassifier(
            n_estimators=10, min_samples_leaf=int(n_samples / 7)
        )

        # rxor rf
        clf_rxor = sklearn.ensemble.RandomForestClassifier(
            n_estimators=10, min_samples_leaf=int(n_samples / 7)
        )

        # train rfs
        # fit the model using the train data
        clf_xor.fit(X_xor_train, y_xor_train)

        # fit rxor model
        clf_rxor.fit(X_rxor_train, y_rxor_train)

        # concat the test samples from xor and rxor (30 from each), 60 total test samples
        X_xor_rxor_test = np.concatenate(
            (X_xor_0[35:], X_rxor_0[35:], X_xor_1[35:], X_rxor_1[35:])
        )
        y_xor_rxor_test = np.concatenate((np.zeros(30), np.ones(30)))

        # predict proba on the new test data with both rfs
        # xor rf
        xor_rxor_test_xorRF_probas = clf_xor.predict_proba(X_xor_rxor_test)

        # rxor rf
        xor_rxor_test_rxorRF_probas = clf_rxor.predict_proba(X_xor_rxor_test)

        # calc the l2 distance between the probas from xor and rxor rfs
        d1 = calcL2(xor_rxor_test_xorRF_probas, xor_rxor_test_rxorRF_probas)

        # concat all xor and rxor samples (100+100=200)
        X_xor_rxor_all = np.concatenate((X_xor, X_rxor))
        y_xor_rxor_all = np.concatenate((y_xor, y_rxor))

        # append the pval
        p_vals.append(
            getPval(X_xor_rxor_all, y_xor_rxor_all, d1, reps, n_samples=n_samples)
        )

    return p_vals


def getPval(X_xor_rxor_all, y_xor_rxor_all, d1, reps=1000, n_samples=100):
    """
    Shuffles xor and rxor, trains trees, predicts, calculates L2 between probas, and calculates p-val to determine whether the 2 distributions are different.
    """
    d1_greater_count = 0
    for i in range(0, reps):
        random_idxs = random.sample(range(200), 200)
        # subsample 100 samples twice randomly, call one xor and the other rxor
        X_xor_new = X_xor_rxor_all[random_idxs[0:100]]
        y_xor_new = y_xor_rxor_all[random_idxs[0:100]]

        X_rxor_new = X_xor_rxor_all[random_idxs[100:]]
        y_rxor_new = y_xor_rxor_all[random_idxs[100:]]

        # subsample 70 from each and call one xor train and one rxor train
        # since we randomly took 100 the pool of 200 samples we should just be able to take the first 70 samples
        X_xor_new_train = X_xor_new[0:70]
        y_xor_new_train = y_xor_new[0:70]

        X_rxor_new_train = X_rxor_new[0:70]
        y_rxor_new_train = y_rxor_new[0:70]

        # train a new forest
        # init the rf's
        # xor rf
        clf_xor_new = sklearn.ensemble.RandomForestClassifier(
            n_estimators=10, min_samples_leaf=int(n_samples / 7)
        )
        clf_xor_new.fit(X_xor_new_train, y_xor_new_train)

        # rxor rf
        clf_rxor_new = sklearn.ensemble.RandomForestClassifier(
            n_estimators=10, min_samples_leaf=int(n_samples / 7)
        )
        clf_rxor_new.fit(X_rxor_new_train, y_rxor_new_train)

        # take the remaing 30 and call those test
        X_xor_new_test = X_xor_new[70:]
        y_xor_new_test = y_xor_new[70:]

        X_rxor_new_test = X_rxor_new[70:]
        y_rxor_new_test = y_rxor_new[70:]

        # concat our new samples
        X_xor_rxor_new_test = np.concatenate((X_xor_new_test, X_rxor_new_test))
        y_xor_rxor_new_test = np.concatenate((y_xor_new_test, y_rxor_new_test))

        # predict proba using the original xor and rxor rf's and calc l2
        # new xor rf
        xor_rxor_new_test_xorRF_probas = clf_xor_new.predict_proba(X_xor_rxor_new_test)

        # new rxor rf
        xor_rxor_new_test_rxorRF_probas = clf_rxor_new.predict_proba(
            X_xor_rxor_new_test
        )

        # calc l2 for our new data
        d2 = calcL2(xor_rxor_new_test_xorRF_probas, xor_rxor_new_test_rxorRF_probas)

        if d1 > d2:
            d1_greater_count += 1

    return 1 - (d1_greater_count / reps)


def calcL2(xorRF_probas, rxorRF_probas):
    """
    Returns L2 distance between 2 outputs from clf.predict_proba().
    """
    # lists to store % label 0 since we only need one of the probas to calc L2
    xors = []
    rxors = []

    # iterate through the passed probas to store them in our lists
    for xor_proba, rxor_proba in zip(xorRF_probas, rxorRF_probas):
        xors.append(xor_proba[0])
        rxors.append(rxor_proba[0])

    return distance.euclidean(xors, rxors)