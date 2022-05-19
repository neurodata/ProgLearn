from proglearn.forest import LifelongClassificationForest
from sdtf import StreamDecisionForest
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from proglearn.sims import generate_gaussian_parity
from math import ceil, log2


def run_gaussian_experiments(mc_rep):
    """
    A function to run both Gaussian R-XOR and XNOR experiments in streaming and batch settings
    """
    # generate all results
    stream_rxor_errors = run_experiment_stream(mc_rep, task2_angle=np.pi / 4)
    stream_xnor_errors = run_experiment_stream(mc_rep, task2_angle=np.pi / 2)
    batch_xnor_error, batch_xnor_le = run_batch_experiment(mc_rep, t2_angle=np.pi / 2)
    batch_rxor_error, batch_rxor_le = run_batch_experiment(mc_rep, t2_angle=np.pi / 4)
    # format for plotting
    streaming_rxor_errors, streaming_rxor_efficiencies = get_learning_efficiencies(
        stream_rxor_errors
    )
    streaming_xnor_errors, streaming_xnor_efficiencies = get_learning_efficiencies(
        stream_xnor_errors
    )
    rxor_results = [
        streaming_rxor_errors,
        streaming_rxor_efficiencies,
        batch_rxor_error,
        batch_rxor_le,
    ]
    xnor_results = [
        streaming_xnor_errors,
        streaming_xnor_efficiencies,
        batch_xnor_error,
        batch_xnor_le,
    ]
    return rxor_results, xnor_results


def run_experiment_stream(mc_rep, task2_angle):
    mean_errors = experiment_stream(task2_angle=task2_angle)
    for i in range(mc_rep - 1):
        mean_errors += experiment_stream(task2_angle=task2_angle)
    mean_errors = mean_errors / mc_rep
    return mean_errors


def get_learning_efficiencies(stream_errors):
    """
    Returns
    ----------
    errors: SynF task 1, Synf task 2, SDF task 1, SDF task 2
    learning_efficiencies: SynF FLE, SDF FLE, SynF BLE, SDF BLE
    """
    stream_synf_FLE = stream_errors[2, :] / stream_errors[6, :]
    sdf_FLE = stream_errors[5, :] / stream_errors[7, :]
    stream_synf_BLE = stream_errors[0, :] / stream_errors[1, :]
    sdf_BLE = stream_errors[3, :] / stream_errors[4, :]
    errors = [stream_errors[1], stream_errors[6], stream_errors[4], stream_errors[7]]
    learning_efficiencies = [stream_synf_FLE, sdf_FLE, stream_synf_BLE, sdf_BLE]
    return errors, learning_efficiencies


def experiment_stream(
    n_task1=750,
    n_task2=750,
    n_update=25,
    n_test=1000,
    task2_angle=np.pi / 2,
    n_trees=10,
):
    """
    A function to do stream SynF and stream decision forest experiment
    between two tasks where the data is generated using Gaussian parity.
    """
    errors = np.zeros((8, ((n_task1 + n_task2) // n_update) - 1), dtype=float)

    # instantiate classifiers
    synf_single_task_t1 = LifelongClassificationForest(default_n_estimators=n_trees)
    synf_multi_task = LifelongClassificationForest(default_n_estimators=n_trees)
    synf_single_task_t2 = LifelongClassificationForest(default_n_estimators=n_trees)
    sdf_single_task_t1 = StreamDecisionForest(n_estimators=n_trees)
    sdf_multi_task = StreamDecisionForest(n_estimators=n_trees)
    sdf_single_task_t2 = StreamDecisionForest(n_estimators=n_trees)

    # generate initial data for add_task
    x1, y1 = generate_gaussian_parity(n_update)
    x2, y2 = generate_gaussian_parity(n_update, angle_params=task2_angle)
    x1_test, y1_test = generate_gaussian_parity(1000)
    x2_test, y2_test = generate_gaussian_parity(1000, angle_params=task2_angle)

    # add tasks to progressive learners/decision forests
    synf_single_task_t1.add_task(x1, y1, task_id=0, classes=[0, 1])
    synf_multi_task.add_task(x1, y1, task_id=0, classes=[0, 1])
    sdf_single_task_t1.partial_fit(x1, y1, classes=[0, 1])
    sdf_multi_task.partial_fit(x1, y1, classes=[0, 1])

    # updating task 1
    for i in range(n_task1 // n_update - 1):
        x, y = generate_gaussian_parity(n_update)
        synf_single_task_t1.update_task(x, y, task_id=0)
        synf_multi_task.update_task(x, y, task_id=0)
        sdf_single_task_t1.partial_fit(x, y)
        sdf_multi_task.partial_fit(x, y)
        synf_t1_y_hat = synf_single_task_t1.predict(x1_test, task_id=0)
        synf_multi_y_hat = synf_multi_task.predict(x1_test, task_id=0)
        sdf_t1_y_hat = sdf_single_task_t1.predict(x1_test)
        sdf_multi_y_hat = sdf_multi_task.predict(x1_test)
        errors[0, i] = 1 - np.mean(synf_t1_y_hat == y1_test)  # synf single task, t1
        errors[1, i] = 1 - np.mean(synf_multi_y_hat == y1_test)  # synf multi task t1
        errors[2, i] = 0.5  # synf single task, t2
        errors[3, i] = 1 - np.mean(sdf_t1_y_hat == y1_test)  # sdf single task, t1
        errors[4, i] = 1 - np.mean(sdf_multi_y_hat == y1_test)  # sdf multi task, t1
        errors[5, i] = 0.5  # sdf single task, t2
        errors[6, i] = 0.5  # synf multi task, t2
        errors[7, i] = 0.5  # sdf multi task, t2

    idx = (n_task1 // n_update) - 1
    # updating task 2
    synf_multi_task.add_task(x2, y2, task_id=1, classes=[0, 1])
    synf_single_task_t2.add_task(x2, y2, task_id=1, classes=[0, 1])
    sdf_single_task_t2.partial_fit(x2, y2, classes=[0, 1])
    sdf_multi_task.partial_fit(x2, y2, classes=[0, 1])

    synf_t1_y_hat = synf_single_task_t1.predict(x1_test, task_id=0)
    synf_t2_y_hat = synf_single_task_t2.predict(x2_test, task_id=1)
    synf_multi_y_hat_t1 = synf_multi_task.predict(x1_test, task_id=0)
    synf_multi_y_hat_t2 = synf_multi_task.predict(x2_test, task_id=1)
    sdf_t1_y_hat = sdf_single_task_t1.predict(x1_test)
    sdf_t2_y_hat = sdf_single_task_t2.predict(x2_test)
    sdf_multi_y_hat_t1 = sdf_multi_task.predict(x1_test)
    sdf_multi_y_hat_t2 = sdf_multi_task.predict(x2_test)

    errors[0, idx] = 1 - np.mean(synf_t1_y_hat == y1_test)  # synf single task, t1
    errors[1, idx] = 1 - np.mean(synf_multi_y_hat_t1 == y1_test)  # synf multi task t1
    errors[2, idx] = 1 - np.mean(synf_t2_y_hat == y2_test)  # synf single task, t2
    errors[3, idx] = 1 - np.mean(sdf_t1_y_hat == y1_test)  # sdf single task, t1
    errors[4, idx] = 1 - np.mean(sdf_multi_y_hat_t1 == y1_test)  # sdf multi task, t1
    errors[5, idx] = 1 - np.mean(sdf_t2_y_hat == y2_test)  # sdf single task, t2
    errors[6, idx] = 1 - np.mean(synf_multi_y_hat_t2 == y2_test)  # synf multi task, t2
    errors[7, idx] = 1 - np.mean(sdf_multi_y_hat_t2 == y2_test)  # sdf multi task, t2
    for i in range(n_task2 // n_update - 1):
        x, y = generate_gaussian_parity(n_update, angle_params=task2_angle)
        synf_multi_task.update_task(x, y, task_id=1)
        synf_single_task_t2.update_task(x, y, task_id=1)
        sdf_single_task_t2.partial_fit(x, y)
        sdf_multi_task.partial_fit(x, y)
        synf_t1_y_hat = synf_single_task_t1.predict(x1_test, task_id=0)
        synf_t2_y_hat = synf_single_task_t2.predict(x2_test, task_id=1)
        synf_multi_y_hat_t1 = synf_multi_task.predict(x1_test, task_id=0)
        synf_multi_y_hat_t2 = synf_multi_task.predict(x2_test, task_id=1)
        sdf_t1_y_hat = sdf_single_task_t1.predict(x1_test)
        sdf_t2_y_hat = sdf_single_task_t2.predict(x2_test)
        sdf_multi_y_hat_t1 = sdf_multi_task.predict(x1_test)
        sdf_multi_y_hat_t2 = sdf_multi_task.predict(x2_test)

        errors[0, i + idx + 1] = 1 - np.mean(
            synf_t1_y_hat == y1_test
        )  # synf single task, t1
        errors[1, i + idx + 1] = 1 - np.mean(
            synf_multi_y_hat_t1 == y1_test
        )  # synf multi task t1
        errors[2, i + idx + 1] = 1 - np.mean(
            synf_t2_y_hat == y2_test
        )  # synf single task, t2
        errors[3, i + idx + 1] = 1 - np.mean(
            sdf_t1_y_hat == y1_test
        )  # sdf single task, t1
        errors[4, i + idx + 1] = 1 - np.mean(
            sdf_multi_y_hat_t1 == y1_test
        )  # sdf multi task, t1
        errors[5, i + idx + 1] = 1 - np.mean(
            sdf_t2_y_hat == y2_test
        )  # sdf single task, t2
        errors[6, i + idx + 1] = 1 - np.mean(
            synf_multi_y_hat_t2 == y2_test
        )  # synf multi task, t2
        errors[7, i + idx + 1] = 1 - np.mean(
            sdf_multi_y_hat_t2 == y2_test
        )  # sdf multi task, t2

    return errors


def experiment_batch(
    n_task1=750,
    n_task2=750,
    n_test=1000,
    task1_angle=0,
    task2_angle=np.pi / 2,
    n_trees=10,
    max_depth=None,
    random_state=None,
):

    """
    A function to do SynF experiment between two tasks
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


def run_batch_experiment(mc_rep, t2_angle):
    n_test = 1000
    n_trees = 10
    n_xor = (100 * np.arange(0.25, 7.50, step=0.25)).astype(int)
    n_xnor = (100 * np.arange(0.25, 7.75, step=0.25)).astype(int)
    mean_error = np.zeros((6, len(n_xor) + len(n_xnor)))
    std_error = np.zeros((6, len(n_xor) + len(n_xnor)))
    mean_te = np.zeros((4, len(n_xor) + len(n_xnor)))
    std_te = np.zeros((4, len(n_xor) + len(n_xnor)))
    for i, n1 in enumerate(n_xor):
        # run experiment in parallel
        error = np.array(
            Parallel(n_jobs=1, verbose=0)(
                delayed(experiment_batch)(
                    n1, 0, task2_angle=t2_angle, max_depth=ceil(log2(n1))
                )
                for _ in range(mc_rep)
            )
        )
        # extract relevant data and store in arrays
        mean_error[:, i] = np.mean(error, axis=0)
        std_error[:, i] = np.std(error, ddof=1, axis=0)
        mean_te[0, i] = np.mean(error[:, 0]) / np.mean(error[:, 1])
        mean_te[1, i] = np.mean(error[:, 2]) / np.mean(error[:, 3])
        mean_te[2, i] = np.mean(error[:, 0]) / np.mean(error[:, 4])
        mean_te[3, i] = np.mean(error[:, 2]) / np.mean(error[:, 5])

        # initialize learning on n-xor data
        if n1 == n_xor[-1]:
            for j, n2 in enumerate(n_xnor):
                # run experiment in parallel
                error = np.array(
                    Parallel(n_jobs=1, verbose=0)(
                        delayed(experiment_batch)(
                            n1, n2, task2_angle=t2_angle, max_depth=ceil(log2(750))
                        )
                        for _ in range(mc_rep)
                    )
                )
                # extract relevant data and store in arrays
                mean_error[:, i + j + 1] = np.mean(error, axis=0)
                std_error[:, i + j + 1] = np.std(error, ddof=1, axis=0)
                mean_te[0, i + j + 1] = np.mean(error[:, 0]) / np.mean(error[:, 1])
                mean_te[1, i + j + 1] = np.mean(error[:, 2]) / np.mean(error[:, 3])
                mean_te[2, i + j + 1] = np.mean(error[:, 0]) / np.mean(error[:, 4])
                mean_te[3, i + j + 1] = np.mean(error[:, 2]) / np.mean(error[:, 5])

    return mean_error, mean_te


def plot_error(results, experiment):
    """Plot Generalization Errors for experiment type (RXOR or XNOR)"""
    algorithms = [
        "Stream Synergistic Forest",
        "Stream Decision Forest",
        "Batch Synergistic Forest",
        "Batch Decision Forest",
    ]
    fontsize = 30
    labelsize = 28
    ls = ["-", "--"]
    colors = sns.color_palette("bright")
    fig = plt.figure(figsize=(26, 14))
    gs = fig.add_gridspec(14, 26)
    ax1 = fig.add_subplot(gs[7:, :5])
    # Stream Synergistic Forest XOR
    ax1.plot(
        (100 * np.arange(0.25, 15, step=0.25)).astype(int),
        results[0][0],
        label=algorithms[0],
        c=colors[3],
        ls=ls[1],
        lw=3,
    )
    # Stream Decision Forest XOR
    ax1.plot(
        (100 * np.arange(0.25, 15, step=0.25)).astype(int),
        results[0][2],
        label=algorithms[1],
        c=colors[2],
        ls=ls[1],
        lw=3,
    )
    # Batch Synergistic Forest XOR
    ax1.plot(
        (100 * np.arange(0.25, 15, step=0.25)).astype(int),
        results[2][1],
        label=algorithms[2],
        c=colors[3],
        ls=ls[0],
        lw=3,
    )
    # Batch Decision Forest XOR
    ax1.plot(
        (100 * np.arange(0.25, 15, step=0.25)).astype(int),
        results[2][4],
        label=algorithms[3],
        c=colors[2],
        ls=ls[0],
        lw=3,
    )

    ax1.set_ylabel("Generalization Error (XOR)", fontsize=fontsize)
    ax1.set_xlabel("Total Sample Size", fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.set_yticks([0.1, 0.3, 0.5, 0.9])
    ax1.set_xticks([0, 750, 1500])
    ax1.axvline(x=750, c="gray", linewidth=1.5, linestyle="dashed")
    ax1.axvline(x=1500, c="gray", linewidth=1.5, linestyle="dashed")

    right_side = ax1.spines["right"]
    right_side.set_visible(False)
    top_side = ax1.spines["top"]
    top_side.set_visible(False)

    ax1.text(200, np.mean(ax1.get_ylim()) + 0.5, "XOR", fontsize=26)
    ax1.text(850, np.mean(ax1.get_ylim()) + 0.5, experiment, fontsize=26)

    ######## RXOR
    ax1 = fig.add_subplot(gs[7:, 7:12])
    rxor_range = (100 * np.arange(0.25, 15, step=0.25)).astype(int)[30:]
    # Stream Synergistic Forest RXOR
    ax1.plot(
        rxor_range,
        results[0][1][30:],
        label=algorithms[0],
        c=colors[3],
        ls=ls[1],
        lw=3,
    )
    # Stream Decision Forest RXOR
    ax1.plot(
        rxor_range,
        results[0][3][30:],
        label=algorithms[1],
        c=colors[2],
        ls=ls[1],
        lw=3,
    )
    # Batch Synergistic Forest RXOR
    ax1.plot(
        rxor_range,
        results[2][3][30:],
        label=algorithms[2],
        c=colors[3],
        ls=ls[0],
        lw=3,
    )
    # Batch Decision Forest RXOR
    ax1.plot(
        rxor_range,
        results[2][5][30:],
        label=algorithms[3],
        c=colors[2],
        ls=ls[0],
        lw=3,
    )
    ax1.set_ylabel("Generalization Error (%s)" % experiment, fontsize=fontsize)
    ax1.legend(
        bbox_to_anchor=(1, -0.25),
        loc="upper center",
        fontsize=20,
        ncol=4,
        frameon=False,
    )
    ax1.set_xlabel("Total Sample Size", fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.set_yticks([0.1, 0.3, 0.5, 0.9])
    ax1.set_xticks([0, 750, 1500])
    ax1.axvline(x=750, c="gray", linewidth=1.5, linestyle="dashed")
    ax1.axvline(x=1500, c="gray", linewidth=1.5, linestyle="dashed")
    right_side = ax1.spines["right"]
    right_side.set_visible(False)
    top_side = ax1.spines["top"]
    top_side.set_visible(False)

    ax1.text(200, np.mean(ax1.get_ylim()) + 0.5, "XOR", fontsize=26)
    ax1.text(850, np.mean(ax1.get_ylim()) + 0.5, experiment, fontsize=26)

    #### Transfer Efficiency
    ax1 = fig.add_subplot(gs[7:, 14:19])
    algorithms = [
        "Stream Synergistic Forest TE",
        "Stream Decision Forest TE",
        "Batch Synergistic Forest TE",
        "Batch Decision Forest TE",
    ]
    rxor_range = (100 * np.arange(0.25, 15, step=0.25)).astype(int)[30:]
    # Stream Synergistic Forest RXOR
    ax1.plot(
        rxor_range,
        np.log(results[1][0][30:]),
        label=algorithms[0],
        c=colors[3],
        ls=ls[1],
        lw=3,
    )
    # Stream Decision Forest RXOR
    ax1.plot(
        rxor_range,
        np.log(results[1][1][30:]),
        label=algorithms[1],
        c=colors[2],
        ls=ls[1],
        lw=3,
    )
    # Batch Synergistic Forest RXOR
    ax1.plot(
        rxor_range,
        np.log(results[3][1][30:]),
        label=algorithms[2],
        c=colors[3],
        ls=ls[0],
        lw=3,
    )
    # Batch Decision Forest RXOR
    ax1.plot(
        rxor_range,
        np.log(results[3][3][30:]),
        label=algorithms[3],
        c=colors[2],
        ls=ls[0],
        lw=3,
    )

    ax1.set_ylabel("Log Forward Learning Efficiency", fontsize=fontsize)
    ax1.set_xlabel("Total Sample Size", fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)
    # ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.set_yticks([-1, 0, 1])
    ax1.set_xlim(-1, 1)
    ax1.set_xticks([0, 750, 1500])
    ax1.axvline(x=750, c="gray", linewidth=1.5, linestyle="dashed")
    ax1.axvline(x=1500, c="gray", linewidth=1.5, linestyle="dashed")
    right_side = ax1.spines["right"]
    right_side.set_visible(False)
    top_side = ax1.spines["top"]
    top_side.set_visible(False)
    ax1.axhline(y=0, c="gray", linewidth=1.5, linestyle="dashed")

    if experiment == "XNOR":
        ax1.text(200, np.mean(ax1.get_ylim()) + 2, "XOR", fontsize=26)
        ax1.text(850, np.mean(ax1.get_ylim()) + 2, experiment, fontsize=26)
    else:
        ax1.text(200, np.mean(ax1.get_ylim()) + 1.25, "XOR", fontsize=26)
        ax1.text(850, np.mean(ax1.get_ylim()) + 1.25, experiment, fontsize=26)

    #### BACKWARDS Transfer Efficiency
    ax1 = fig.add_subplot(gs[7:, 21:])
    algorithms = [
        "Stream Synergistic Forest TE",
        "Stream Decision Forest TE",
        "Batch Synergistic Forest TE",
        "Batch Decision Forest TE",
    ]
    rxor_range = (100 * np.arange(0.25, 15, step=0.25)).astype(int)
    # Stream Synergistic Forest RXOR
    ax1.plot(
        rxor_range,
        np.log(results[1][2]),
        label=algorithms[0],
        c=colors[3],
        ls=ls[1],
        lw=3,
    )
    # Stream Decision Forest RXOR
    ax1.plot(
        rxor_range,
        np.log(results[1][3]),
        label=algorithms[1],
        c=colors[2],
        ls=ls[1],
        lw=3,
    )
    # Batch Synergistic Forest RXOR
    ax1.plot(
        rxor_range,
        np.log(results[3][0]),
        label=algorithms[2],
        c=colors[3],
        ls=ls[0],
        lw=3,
    )
    # Batch Decision Forest RXOR
    ax1.plot(
        rxor_range,
        np.log(results[3][2]),
        label=algorithms[3],
        c=colors[2],
        ls=ls[0],
        lw=3,
    )

    ax1.set_ylabel("Log Backward Learning Efficiency", fontsize=fontsize)
    ax1.set_xlabel("Total Sample Size", fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)
    # ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.set_yticks([-1, 0, 1])
    ax1.set_xlim(-1, 1)
    ax1.set_xticks([25, 750, 1500])
    ax1.axvline(x=750, c="gray", linewidth=1.5, linestyle="dashed")
    ax1.axvline(x=1500, c="gray", linewidth=1.5, linestyle="dashed")
    right_side = ax1.spines["right"]
    right_side.set_visible(False)
    top_side = ax1.spines["top"]
    top_side.set_visible(False)
    ax1.axhline(y=0, c="gray", linewidth=1.5, linestyle="dashed")

    if experiment == "XNOR":
        ax1.text(200, np.mean(ax1.get_ylim()) + 2.0, "XOR", fontsize=26)
        ax1.text(850, np.mean(ax1.get_ylim()) + 2.0, experiment, fontsize=26)
    else:
        ax1.text(200, np.mean(ax1.get_ylim()) + 1.5, "XOR", fontsize=26)
        ax1.text(850, np.mean(ax1.get_ylim()) + 1.5, experiment, fontsize=26)
