import numpy as np


def generate_spirals(
    n_samples,
    n_class=2,
    noise=0.3,
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

    n_class : array of shape [n_centers], optional (default=2)
        Number of class for the spiral simulation.

    noise : float, optional (default=0.3)
        Parameter controlling the spread of each class.

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

    X = []
    y = []

    if n_class == 2:
        turns = 2
    elif n_class == 3:
        turns = 2.5
    elif n_class == 5:
        turns = 3.5
    elif n_class == 7:
        turns = 4.5
    else:
        raise ValueError("sorry, can't currently surpport %s classes " % n_class)

    mvt = np.random.multinomial(n_samples, 1 / n_class * np.ones(n_class))

    if n_class == 2:
        r = np.random.uniform(0, 1, size=int(n_samples / n_class))
        r = np.sort(r)
        t = np.linspace(
            0, np.pi * 4 * turns / n_class, int(n_samples / n_class)
        ) + np.random.normal(0, noise, int(n_samples / n_class))
        dx = r * np.cos(t)
        dy = r * np.sin(t)

        X.append(np.vstack([dx, dy]).T)
        X.append(np.vstack([-dx, -dy]).T)
        y += [0] * int(n_samples / n_class)
        y += [1] * int(n_samples / n_class)
    else:
        for j in range(1, n_class + 1):
            r = np.linspace(0.01, 1, int(mvt[j - 1]))
            t = (
                np.linspace(
                    (j - 1) * np.pi * 4 * turns / n_class,
                    j * np.pi * 4 * turns / n_class,
                    int(mvt[j - 1]),
                )
                + np.random.normal(0, noise, int(mvt[j - 1]))
            )

            dx = r * np.cos(t)
            dy = r * np.sin(t)

            dd = np.vstack([dx, dy]).T
            X.append(dd)
            y += [j - 1] * int(mvt[j - 1])

    return np.vstack(X), np.array(y).astype(int)
