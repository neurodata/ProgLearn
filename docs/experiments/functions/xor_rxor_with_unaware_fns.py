# import
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# k sample testing from hyppo
from hyppo.ksample import KSample

from proglearn.forest import LifelongClassificationForest
from math import log2, ceil
from joblib import Parallel, delayed
import seaborn as sns


def generate_gaussian_parity(
    n_samples,
    centers=None,
    class_label=None,
    cluster_std=0.25,
    center_box=(-1.0, 1.0),
    angle_params=None,
    random_state=None,
):
    """
    Generate 2-dimensional Gaussian XOR distribution.
    (Classic XOR problem but each point is the
    center of a Gaussian blob distribution)
    Parameters
    ----------
    n_samples : int
        Total number of points divided among the four
        clusters with equal probability.
    centers : array of shape [n_centers,2], optional (default=None)
        The coordinates of the ceneter of total n_centers blobs.
    class_label : array of shape [n_centers], optional (default=None)
        class label for each blob.
    cluster_std : float, optional (default=1)
        The standard deviation of the blobs.
    center_box : tuple of float (min, max), default=(-1.0, 1.0)
        The bounding box for each cluster center when centers are generated at random.
    angle_params: float, optional (default=None)
        Number of radians to rotate the distribution by.
    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """

    if random_state != None:
        np.random.seed(random_state)

    if centers == None:
        centers = np.array([(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)])

    if class_label == None:
        class_label = [0, 1, 1, 0]

    blob_num = len(class_label)

    # get the number of samples in each blob with equal probability
    samples_per_blob = np.random.multinomial(
        n_samples, 1 / blob_num * np.ones(blob_num)
    )

    X, y = make_blobs(
        n_samples=samples_per_blob,
        n_features=2,
        centers=centers,
        cluster_std=cluster_std,
        center_box=center_box,
    )

    for blob in range(blob_num):
        y[np.where(y == blob)] = class_label[blob]

    if angle_params != None:
        R = _generate_2d_rotation(angle_params)
        X = X @ R

    return X, y.astype(int)


def _generate_2d_rotation(theta=0):
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    return R


def calc_ksample_pval_vs_angle(mc_reps, angle_sweep):
    last_angle = max(angle_sweep) + 1

    # arrays to store stats and pvals, 100 samples
    stats_100 = np.empty([last_angle, mc_reps])
    pvals_100 = np.empty([last_angle, mc_reps])

    # arrays to store stats and pvals. 500 samples
    stats_500 = np.empty([last_angle, mc_reps])
    pvals_500 = np.empty([last_angle, mc_reps])

    # arrays to store stats and pvals, 1000 samples
    stats_1000 = np.empty([last_angle, mc_reps])
    pvals_1000 = np.empty([last_angle, mc_reps])

    # run exp 10 times
    for k in range(mc_reps):
        for n in angle_sweep:
            # 100 samples
            n_samples = 100
            xor = generate_gaussian_parity(n_samples, random_state=k)
            rxor = generate_gaussian_parity(
                n_samples, angle_params=np.radians(n), random_state=k
            )
            stats_100[n, k], pvals_100[n, k] = KSample(indep_test="Dcorr").test(
                xor[0], rxor[0]
            )

            # 500 samples
            n_samples = 500
            xor = generate_gaussian_parity(n_samples, random_state=k)
            rxor = generate_gaussian_parity(
                n_samples, angle_params=np.radians(n), random_state=k
            )
            stats_500[n, k], pvals_500[n, k] = KSample(indep_test="Dcorr").test(
                xor[0], rxor[0]
            )

            # 1000 samples
            n_samples = 1000
            xor = generate_gaussian_parity(n_samples, random_state=k)
            rxor = generate_gaussian_parity(
                n_samples, angle_params=np.radians(n), random_state=k
            )
            stats_1000[n, k], pvals_1000[n, k] = KSample(indep_test="Dcorr").test(
                xor[0], rxor[0]
            )

    return pvals_100, pvals_500, pvals_1000


def plot_pval_vs_angle(pvals_100, pvals_500, pvals_1000, angle_sweep):
    # avg the experiments
    pval_means_100 = np.mean(pvals_100, axis=1)
    pval_means_500 = np.mean(pvals_500, axis=1)
    pval_means_1000 = np.mean(pvals_1000, axis=1)

    # add error bars
    qunatiles_100 = np.nanquantile(pvals_100, [0.25, 0.75], axis=1)
    qunatiles_500 = np.nanquantile(pvals_500, [0.25, 0.75], axis=1)
    qunatiles_1000 = np.nanquantile(pvals_1000, [0.25, 0.75], axis=1)
    plt.fill_between(
        angle_sweep, qunatiles_100[0], qunatiles_100[1], facecolor="r", alpha=0.3
    )
    plt.fill_between(
        angle_sweep, qunatiles_500[0], qunatiles_500[1], facecolor="b", alpha=0.3
    )
    plt.fill_between(
        angle_sweep, qunatiles_1000[0], qunatiles_1000[1], facecolor="cyan", alpha=0.3
    )

    # plot
    plt.xlabel("Angle of Rotation")
    plt.ylabel("P-Value")
    plt.title("Angle of Rotation vs K-sample Test P-Value")
    plt.plot(angle_sweep, pval_means_100, label="100 samples", color="r")
    plt.plot(angle_sweep, pval_means_500, label="500 samples", color="b")
    plt.plot(angle_sweep, pval_means_1000, label="1000 samples", color="cyan")

    # draw line at p val = 0.05
    plt.axhline(y=0.05, color="g", linestyle="--")

    plt.show
    plt.legend()


# calc BTE and gen error, runs aware_experiment function
# modified from xor/rxor experiment in proglearn experiments folder
def bte_ge_v_angle(angle_sweep, task1_sample, task2_sample, mc_rep):
    mean_te = np.zeros(len(angle_sweep), dtype=float)
    mean_error = np.zeros([len(angle_sweep), 6], dtype=float)
    for ii, angle in enumerate(angle_sweep):
        error = np.array(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(aware_experiment)(
                    task1_sample,
                    task2_sample,
                    task2_angle=angle * np.pi / 180,
                    max_depth=ceil(log2(task1_sample)),
                )
                for _ in range(mc_rep)
            )
        )

        mean_te[ii] = np.mean(error[:, 0]) / np.mean(error[:, 1])
        mean_error[ii] = np.mean(error, axis=0)
    return mean_te, mean_error


# below function from xor/rxor experiment in proglearn experiments folder
def aware_experiment(
    n_task1,
    n_task2,
    n_test=1000,
    task1_angle=0,
    task2_angle=np.pi / 2,
    n_trees=10,
    max_depth=None,
    random_state=None,
):

    """
    A function to do Odif experiment between two tasks
    where the task data is generated using Gaussian parity.
    Parameters
    ----------
    n_task1 : int
        Total number of train sample for task 1.
    n_task2 : int
        Total number of train dsample for task 2
    n_test : int, optional (default=1000)
        Number of test sample for each task.
    task1_angle : float, optional (default=0)
        Angle in radian for task 1.
    task2_angle : float, optional (default=numpy.pi/2)
        Angle in radian for task 2.
    n_trees : int, optional (default=10)
        Number of total trees to train for each task.
    max_depth : int, optional (default=None)
        Maximum allowable depth for each tree.
    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    Returns
    -------
    errors : array of shape [6]
        Elements of the array is organized as single task error task1,
        multitask error task1, single task error task2,
        multitask error task2, naive UF error task1,
        naive UF task2.
    """

    if n_task1 == 0 and n_task2 == 0:
        raise ValueError("Wake up and provide samples to train!!!")

    if random_state != None:
        np.random.seed(random_state)

    errors = np.zeros(6, dtype=float)

    progressive_learner = LifelongClassificationForest(default_n_estimators=n_trees)
    uf1 = LifelongClassificationForest(default_n_estimators=n_trees)
    naive_uf = LifelongClassificationForest(default_n_estimators=n_trees)
    uf2 = LifelongClassificationForest(default_n_estimators=n_trees)

    # source data
    X_task1, y_task1 = generate_gaussian_parity(n_task1, angle_params=task1_angle)
    test_task1, test_label_task1 = generate_gaussian_parity(
        n_test, angle_params=task1_angle
    )

    # target data
    X_task2, y_task2 = generate_gaussian_parity(n_task2, angle_params=task2_angle)
    test_task2, test_label_task2 = generate_gaussian_parity(
        n_test, angle_params=task2_angle
    )

    if n_task1 == 0:
        progressive_learner.add_task(X_task2, y_task2, n_estimators=n_trees)
        uf2.add_task(X_task2, y_task2, n_estimators=n_trees)

        errors[0] = 0.5
        errors[1] = 0.5

        uf_task2 = uf2.predict(test_task2, task_id=0)
        l2f_task2 = progressive_learner.predict(test_task2, task_id=0)

        errors[2] = 1 - np.mean(uf_task2 == test_label_task2)
        errors[3] = 1 - np.mean(l2f_task2 == test_label_task2)

        errors[4] = 0.5
        errors[5] = 1 - np.mean(uf_task2 == test_label_task2)
    elif n_task2 == 0:
        progressive_learner.add_task(X_task1, y_task1, n_estimators=n_trees)
        uf1.add_task(X_task1, y_task1, n_estimators=n_trees)

        uf_task1 = uf1.predict(test_task1, task_id=0)
        l2f_task1 = progressive_learner.predict(test_task1, task_id=0)

        errors[0] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[1] = 1 - np.mean(l2f_task1 == test_label_task1)

        errors[2] = 0.5
        errors[3] = 0.5

        errors[4] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[5] = 0.5
    else:
        progressive_learner.add_task(X_task1, y_task1, n_estimators=n_trees)
        progressive_learner.add_task(X_task2, y_task2, n_estimators=n_trees)

        uf1.add_task(X_task1, y_task1, n_estimators=2 * n_trees)
        uf2.add_task(X_task2, y_task2, n_estimators=2 * n_trees)

        naive_uf_train_x = np.concatenate((X_task1, X_task2), axis=0)
        naive_uf_train_y = np.concatenate((y_task1, y_task2), axis=0)
        naive_uf.add_task(naive_uf_train_x, naive_uf_train_y, n_estimators=n_trees)

        uf_task1 = uf1.predict(test_task1, task_id=0)
        l2f_task1 = progressive_learner.predict(test_task1, task_id=0)
        uf_task2 = uf2.predict(test_task2, task_id=0)
        l2f_task2 = progressive_learner.predict(test_task2, task_id=1)
        naive_uf_task1 = naive_uf.predict(test_task1, task_id=0)
        naive_uf_task2 = naive_uf.predict(test_task2, task_id=0)

        errors[0] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[1] = 1 - np.mean(l2f_task1 == test_label_task1)
        errors[2] = 1 - np.mean(uf_task2 == test_label_task2)
        errors[3] = 1 - np.mean(l2f_task2 == test_label_task2)
        errors[4] = 1 - np.mean(naive_uf_task1 == test_label_task1)
        errors[5] = 1 - np.mean(naive_uf_task2 == test_label_task2)

    return errors


# below function from xor/rxor experiment in proglearn experiments folder
def plot_bte_v_angle(mean_te):
    angle_sweep = range(0, 90, 1)

    sns.set_context("talk")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(angle_sweep, mean_te, linewidth=3, c="r")
    ax.set_xticks([0, 45, 90])
    ax.set_xlabel("Angle of Rotation (Degrees)")
    ax.set_ylabel("Backward Transfer Efficiency (XOR)")
    ax.hlines(1, 0, 90, colors="gray", linestyles="dashed", linewidth=1.5)

    ax.set_yticks([0.9, 1, 1.1, 1.2])
    ax.set_ylim(0.89, 1.22)
    log_lbl = np.round(np.log([0.9, 1, 1.1, 1.2]), 2)
    labels = [item.get_text() for item in ax.get_yticklabels()]

    for ii, _ in enumerate(labels):
        labels[ii] = str(log_lbl[ii])

    ax.set_yticklabels(labels)

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)


def unaware_experiment(
    n_task1,
    n_task2,
    n_test=1000,
    task1_angle=0,
    task2_angle=np.pi / 2,
    n_trees=10,
    max_depth=None,
    random_state=None,
):

    """
    A function to do Odif experiment between two tasks
    where the task data is generated using Gaussian parity.
    Parameters
    ----------
    n_task1 : int
        Total number of train sample for task 1.
    n_task2 : int
        Total number of train dsample for task 2
    n_test : int, optional (default=1000)
        Number of test sample for each task.
    task1_angle : float, optional (default=0)
        Angle in radian for task 1.
    task2_angle : float, optional (default=numpy.pi/2)
        Angle in radian for task 2.
    n_trees : int, optional (default=10)
        Number of total trees to train for each task.
    max_depth : int, optional (default=None)
        Maximum allowable depth for each tree.
    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    Returns
    -------
    errors : array of shape [6]
        Elements of the array is organized as single task error task1,
        multitask error task1, single task error task2,
        multitask error task2, naive UF error task1,
        naive UF task2.
    """

    if n_task1 == 0 and n_task2 == 0:
        raise ValueError("Wake up and provide samples to train!!!")

    if random_state != None:
        np.random.seed(random_state)

    errors = np.zeros(6, dtype=float)

    progressive_learner = LifelongClassificationForest(default_n_estimators=n_trees)
    uf1 = LifelongClassificationForest(default_n_estimators=n_trees)
    naive_uf = LifelongClassificationForest(default_n_estimators=n_trees)
    uf2 = LifelongClassificationForest(default_n_estimators=n_trees)

    # source data
    X_task1, y_task1 = generate_gaussian_parity(n_task1, angle_params=task1_angle)
    test_task1, test_label_task1 = generate_gaussian_parity(
        n_test, angle_params=task1_angle
    )

    # target data
    X_task2, y_task2 = generate_gaussian_parity(n_task2, angle_params=task2_angle)
    test_task2, test_label_task2 = generate_gaussian_parity(
        n_test, angle_params=task2_angle
    )

    if KSample(indep_test="Dcorr").test(X_task1, X_task2)[1] <= 0.05:
        progressive_learner.add_task(X_task1, y_task1, n_estimators=n_trees)
        progressive_learner.add_task(X_task2, y_task2, n_estimators=n_trees)

        uf1.add_task(X_task1, y_task1, n_estimators=2 * n_trees)
        uf2.add_task(X_task2, y_task2, n_estimators=2 * n_trees)

        naive_uf_train_x = np.concatenate((X_task1, X_task2), axis=0)
        naive_uf_train_y = np.concatenate((y_task1, y_task2), axis=0)
        naive_uf.add_task(naive_uf_train_x, naive_uf_train_y, n_estimators=n_trees)

        uf_task1 = uf1.predict(test_task1, task_id=0)
        l2f_task1 = progressive_learner.predict(test_task1, task_id=0)
        uf_task2 = uf2.predict(test_task2, task_id=0)
        l2f_task2 = progressive_learner.predict(test_task2, task_id=1)
        naive_uf_task1 = naive_uf.predict(test_task1, task_id=0)
        naive_uf_task2 = naive_uf.predict(test_task2, task_id=0)

        errors[0] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[1] = 1 - np.mean(l2f_task1 == test_label_task1)
        errors[2] = 1 - np.mean(uf_task2 == test_label_task2)
        errors[3] = 1 - np.mean(l2f_task2 == test_label_task2)
        errors[4] = 1 - np.mean(naive_uf_task1 == test_label_task1)
        errors[5] = 1 - np.mean(naive_uf_task2 == test_label_task2)
    else:
        naive_uf_train_x = np.concatenate((X_task1, X_task2), axis=0)
        naive_uf_train_y = np.concatenate((y_task1, y_task2), axis=0)
        progressive_learner.add_task(
            naive_uf_train_x, naive_uf_train_y, n_estimators=n_trees
        )

        uf1.add_task(X_task1, y_task1, n_estimators=2 * n_trees)
        uf2.add_task(X_task2, y_task2, n_estimators=2 * n_trees)

        naive_uf.add_task(naive_uf_train_x, naive_uf_train_y, n_estimators=n_trees)

        uf_task1 = uf1.predict(test_task1, task_id=0)
        l2f_task1 = progressive_learner.predict(test_task1, task_id=0)
        uf_task2 = uf2.predict(test_task2, task_id=0)
        # l2f_task2 = progressive_learner.predict(test_task2, task_id=1) # NOT USED here but is used in task aware setting
        naive_uf_task1 = naive_uf.predict(test_task1, task_id=0)
        naive_uf_task2 = naive_uf.predict(test_task2, task_id=0)

        errors[0] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[1] = 1 - np.mean(l2f_task1 == test_label_task1)
        errors[2] = 1 - np.mean(uf_task2 == test_label_task2)
        errors[3] = 0
        errors[4] = 1 - np.mean(naive_uf_task1 == test_label_task1)
        errors[5] = 1 - np.mean(naive_uf_task2 == test_label_task2)

    return errors


# modified function from xor/rxor experiment in proglearn experiments
# returns numpy arrays mean_te and mean_error
def unaware_bte_v_angle(angle_sweep, task1_sample, task2_sample, mc_rep):
    mean_te = np.zeros(len(angle_sweep), dtype=float)
    mean_error = np.zeros([len(angle_sweep), 6], dtype=float)
    for ii, angle in enumerate(angle_sweep):
        error = np.array(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(unaware_experiment)(
                    task1_sample,
                    task2_sample,
                    task2_angle=angle * np.pi / 180,
                    max_depth=ceil(log2(task1_sample)),
                )
                for _ in range(mc_rep)
            )
        )

        mean_te[ii] = np.mean(error[:, 0]) / np.mean(error[:, 1])
        mean_error[ii] = np.mean(error, axis=0)
    return mean_te, mean_error


# modified plot bte using function from xor/rxor example
def plot_unaware_bte_v_angle(mean_te):
    angle_sweep = range(0, 90, 1)

    sns.set_context("talk")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(angle_sweep, mean_te, linewidth=3, c="r")
    ax.set_xticks([0, 45, 90])
    ax.set_xlabel("Angle of Rotation (Degrees)")
    ax.set_ylabel("Backward Transfer Efficiency (XOR)")
    ax.hlines(1, 0, 90, colors="gray", linestyles="dashed", linewidth=1.5)

    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2])
    log_lbl = np.round(
        np.log([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]), 2
    )
    labels = [item.get_text() for item in ax.get_yticklabels()]

    for ii, _ in enumerate(labels):
        labels[ii] = str(log_lbl[ii])

    ax.set_yticklabels(labels)

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
