import numpy as np
from sklearn.random_projection import SparseRandomProjection


def random_matrix_binary(X, sample_inds, density, feature_combinations, random_state):
    """
    Gets the projection matrix and it fits the transform to the samples of interest.

    Parameters
    ----------
    X : array of shape [n_samples, n_features]
        The input data X is a matrix of the examples and their respective feature
        values for each of the features.
    sample_inds : array of shape [n_samples]
        The data we are transforming.
    density : float
        Ratio of non-zero component in the random projection matrix in the range '(0, 1]'.
    feature_combinations : float
        The feature combinations to use for the oblique split.
    random_state : int
        Controls the pseudo random number generator used to generate the projection matrix.

    Returns
    -------
    proj_mat : {ndarray, sparse matrix} of shape (self.proj_dims, n_features)
        The generated sparse random matrix.
    proj_X : {ndarray, sparse matrix} of shape (n_samples, self.proj_dims)
        Projected input data matrix.
    """

    proj_dims = int(np.ceil(X.shape[1]) / feature_combinations)
    proj_mat = SparseRandomProjection(
        density=density, n_components=proj_dims, random_state=random_state,
    )
    proj_X = proj_mat.fit_transform(X[sample_inds, :])

    return proj_X, proj_mat
