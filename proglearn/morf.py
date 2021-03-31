import numpy as np
from sklearn.utils import check_random_state
from proglearn.transformers import ObliqueTreeClassifier, ObliqueSplitter
from proglearn.split import BaseObliqueSplitter
import numpy.random as rng


def _check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


class Conv2DSplitter(ObliqueSplitter):
    """Convolutional splitter.

    A class used to represent a 2D convolutional splitter, where splits
    are done on a convolutional patch.

    Note: The convolution function is currently just the
    summation operator.

    Parameters
    ----------
    X : array of shape [n_samples, n_features]
        The input data X is a matrix of the examples and their respective feature
        values for each of the features.
    y : array of shape [n_samples]
        The labels for each of the examples in X.
    max_features : float
        controls the dimensionality of the target projection space.
    feature_combinations : float
        controls the density of the projection matrix
    random_state : int
        Controls the pseudo random number generator used to generate the projection matrix.
    image_height : [type], optional
        [description], by default None
    image_width : [type], optional
        [description], by default None
    patch_height_max : [type], optional
        [description], by default None
    patch_height_min : int, optional
        [description], by default 1
    patch_width_max : [type], optional
        [description], by default None
    patch_width_min : int, optional
        [description], by default 1
    discontiguous_height : bool, optional
        [description], by default False
    discontiguous_width : bool, optional
        [description], by default False

    Methods
    -------
    sample_proj_mat
        Will compute projection matrix, which has columns as the vectorized
        convolutional patches.

    Notes
    -----
    This class assumes that data is vectorized in
    row-major (C-style), rather then column-major (Fotran-style) order.
    """

    def __init__(
        self,
        X,
        y,
        max_features,
        feature_combinations,
        random_state,
        image_height=None,
        image_width=None,
        patch_height_max=None,
        patch_height_min=1,
        patch_width_max=None,
        patch_width_min=1,
        discontiguous_height: bool = False,
        discontiguous_width: bool = False,
        debug: bool = False,
    ):
        super(Conv2DSplitter, self).__init__(
            X=X,
            y=y,
            max_features=max_features,
            feature_combinations=feature_combinations,
            random_state=random_state,
        )

        # Check that image_height and image_width are divisors of
        # the num_features.  This is the most we can do to
        # prevent an invalid value being passed in.
        if (self.n_features % image_height) != 0:
            raise ValueError("Incorrect image_height given:")
        else:
            self.image_height = image_height
        if (self.n_features % image_width) != 0:
            raise ValueError("Incorrect image_width given:")
        else:
            self.image_width = image_width

        # If patch_height_{min, max} and patch_width_{min, max} are
        # not set by the user, set them to defaults.
        if patch_height_max is None:
            self.patch_height_max_ = max(2, np.floor(np.sqrt(self.image_height)))
        else:
            self.patch_height_max = patch_height_max
        if patch_width_max is None:
            self.patch_width_max = max(2, np.floor(np.sqrt(self.image_width)))
        else:
            self.patch_width_max = patch_width_max
        if 1 <= patch_height_min <= self.patch_height_max:
            self.patch_height_min = patch_height_min
        else:
            raise ValueError("Incorrect patch_height_min")
        if 1 <= patch_width_min <= self.patch_width_max:
            self.patch_width_min = patch_width_min
        else:
            raise ValueError("Incorrect patch_width_min")

        # set sample dimensions
        self.axis_sample_dims = [
            (patch_height_min, patch_height_max),
            (patch_width_min, patch_width_max),
        ]
        self.structured_data_shape = [image_height, image_width]
        self.debug = debug

    def _get_rand_patch_idx(self):
        """Generates a random patch on the original data to consider as feature combination.

        This function assumes that data samples were vectorized. Thus contiguous convolutional
        patches are defined based on the top left corner.

        Returns
        -------
        height_width_top : tuple of (height, width, topleft point)
            [description]
        """
        # A vector of vectors that specifies the parameters
        # for each patch: <Height>, <Width>, <TopLeft>
        # height_width_top = []

        # generate a random height and width of the patch
        rand_height = rng.randint(self.patch_height_min, self.patch_height_max)
        rand_width = rng.randint(self.patch_width_min, self.patch_width_max)

        # XXX: results in edge effect on the RHS of the structured data...
        # compute the difference between the image dimension and current random
        # patch dimension
        delta_height = self.image_height - rand_height + 1
        delta_width = self.image_width - rand_width + 1

        # sample the top left pixel from available pixels now
        top_left_point = rng.randint(delta_width * delta_height)

        # convert the top left point to appropriate index in full image
        vectorized_start_idx = int(
            (top_left_point % delta_width)
            + (self.image_width * np.floor(top_left_point / delta_width))
        )

        return (rand_height, rand_width, vectorized_start_idx)

    def _compute_index_in_vectorized_data(self, idx):
        return np.ravel_multi_index(
            idx, dims=self.structured_data_shape, mode="raise", order="C"
        )

    def _compute_vectorized_index_in_data(self, vec_idx):
        return np.unravel_index(vec_idx, shape=self.structured_data_shape, order="C")

    def sample_proj_mat(self, sample_inds):
        """
        Gets the projection matrix and it fits the transform to the samples of interest.

        Parameters
        ----------
        sample_inds : array of shape [n_samples]
            The data we are transforming.

        Returns
        -------
        proj_mat : {ndarray, sparse matrix} of shape (self.proj_dims, n_features)
            The generated sparse random matrix.
        proj_X : {ndarray, sparse matrix} of shape (n_samples, self.proj_dims)
            Projected input data matrix.

        Notes
        -----
        See `randMatTernary` in rerf.py for SPORF.

        See `randMat
        """
        # creates a projection matrix where columns are vectorized patch
        # combinations
        proj_mat = np.zeros((self.n_features, self.proj_dims))

        # loop over mtry to load random patch dimensions and the
        # top left position
        # Note: max_features is aka mtry
        for idx in range(self.proj_dims):
            # get patch positions
            height_width_top = self._get_rand_patch_idx()
            rand_height, rand_width, vectorized_start_idx = height_width_top

            # get the (x_1, x_2) coordinate in 2D array of sample
            multi_idx = self._compute_vectorized_index_in_data(vectorized_start_idx)

            if self.debug:
                print(idx, vectorized_start_idx, multi_idx, rand_height, rand_width)
            structured_patch_idxs = np.ix_(
                np.arange(multi_idx[0], multi_idx[0] + rand_height),
                np.arange(multi_idx[1], multi_idx[1] + rand_width),
            )

            # get the patch vectorized indices
            patch_idxs = self._compute_index_in_vectorized_data(structured_patch_idxs)

            # get indices for this patch
            proj_mat[patch_idxs, idx] = 1

        proj_X = self.X[sample_inds, :] @ proj_mat
        return proj_X, proj_mat


# XXX: refactor to: convolutional splitter, graph splitter
class MorfObliqueSplitter(ObliqueSplitter):
    """
    A class used to represent a manifold oblique splitter, where splits are done on
    the linear combination of the features. This class of splitter
    overrides the sampled projection matrix.

    The manifolds in consideration are:

    1. multivariate data: X is of the form (X1, X2, ...), where each Xi is of
    dimension d(i).
    2. graph data: X is of the form G, where G is a class of graphs.

    There are two ways of

    Parameters
    ----------
    X : array of shape [n_samples, n_features]
        The input data X is a matrix of the examples and their respective feature
        values for each of the features.
    y : array of shape [n_samples]
        The labels for each of the examples in X.
    max_features : float
        controls the dimensionality of the target projection space.
    feature_combinations : float
        controls the density of the projection matrix
    random_state : int
        Controls the pseudo random number generator used to generate the projection matrix.
    n_jobs : int
        The number of cores to parallelize the calculation of Gini impurity.
        Supply -1 to use all cores available to the Process.

    Methods
    -------
    _compute_vectorized_index_in_data(vec_idx)
        Will compute the corresponding index in the data based on
        an index chosen in the vectorized data.

    Notes
    -----
    This class assumes that data is vectorized in
    row-major (C-style), rather then column-major (Fotran-style) order.
    """

    def __init__(
        self,
        X,
        y,
        axis_sample_dims: list,
        max_features,
        feature_combinations,
        random_state,
        n_jobs,
        axis_sample_graphs: list = None,
        axis_data_dims: list = None,
    ):
        super(ManifoldObliqueTreeClassifier, self).__init__(
            X=X,
            y=y,
            max_features=max_features,
            feature_combinations=feature_combinations,
            random_state=random_state,
            workers=n_jobs,
        )

        if axis_sample_graphs is None and axis_data_dims is None:
            raise RuntimeError(
                "Either the sample graph must be instantiated, or "
                "the data dimensionality must be specified. Both are not right now."
            )

        # error check sampling graphs
        if axis_sample_graphs is not None:
            # perform error check on the passes in sample graphs and dimensions
            if len(axis_sample_graphs) != len(axis_sample_dims):
                raise ValueError(
                    f"The number of sample graphs \
                ({len(axis_sample_graphs)}) must match \
                the number of sample dimensions ({len(axis_sample_dims)}) in MORF."
                )
            if not all([x.ndim == 2 for x in axis_sample_graphs]):
                raise ValueError(
                    f"All axis sample graphs must be \
                                    2D matrices."
                )
            if not all([x.shape[0] == x.shape[1] for x in axis_sample_graphs]):
                raise ValueError(f"All axis sample graphs must be " "square matrices.")

            # XXX: could later generalize to remove this condition
            if not all([_check_symmetric(x) for x in axis_sample_graphs]):
                raise ValueError("All axis sample graphs must" "be symmetric.")

        # error check data dimensions
        if axis_data_dims is not None:
            # perform error check on the passes in sample graphs and dimensions
            if len(axis_data_dims) != len(axis_sample_dims):
                raise ValueError(
                    f"The number of data dimensions "
                    "({len(axis_data_dims)}) must match "
                    "the number of sample dimensions ({len(axis_sample_dims)}) in MORF."
                )

            if X.shape[1] != np.sum(axis_data_dims):
                raise ValueError(
                    f"The specified data dimensionality "
                    "({np.sum(axis_data_dims)}) does not match the dimensionality "
                    "of the data (i.e. # columns in X: {X.shape[1]})."
                )

        self.axis_sample_dims = axis_sample_dims
        self.axis_sample_graphs = axis_sample_graphs
        self.structured_data_shape = []

        if axis_sample_graphs is not None:
            for graph in self.axis_sample_graphs:
                self.structured_data_shape.append(graph.shape[0])
        else:
            self.structured_data_shape.extend(axis_data_dims)

        self.proj_dims = int(max_features * self.n_features)

    def _compute_index_in_vectorized_data(self, idx):
        return np.ravel_multi_index(
            idx, dims=self.structured_data_shape, mode="raise", order="C"
        )

    def _compute_vectorized_index_in_data(self, vec_idx):
        return np.unravel_index(vec_idx, shape=self.structured_data_shape, order="C")

    def _get_rand_patch_idx(self):
        """Generates"""
        rnd_mtry = 0
        rnd_feature = 0
        rnd_weight = 0

        # stores vectorized_pivot_point corresponding to patches to take
        patch_vectorized_indices = []

        # stores the dimensions of corresponding patches to take about the
        # vectorized feature point
        patch_dims = []

        # loop over mtry to load random patch dimensions and the
        # top left position
        # Note: max_features is aka mtry
        for idx in range(self.max_features):
            # pick random patch size for each axis of data
            axis_sample_lens = []
            delta_axis_lens = []
            axis_start_idx = []

            for idx, (axis_min_len, axis_max_len) in enumerate(self.axis_sample_dims):
                # sample from [axis_patchMin, axis_patchMax]
                axis_sample_len = rng.randint(low=axis_min_len, high=axis_max_len)
                axis_sample_lens.append(axis_sample_len)

                # compute area that sample can occur at
                delta_len = self.structured_data_shape[idx] - axis_sample_len
                delta_axis_lens.append(delta_len)

                # compute random position in this axis to generate the pivot point
                this_axis_pivot_idx = rng.randint(0, delta_len)
                axis_start_idx.append(this_axis_pivot_idx)

            # compute the vectorized points at which these patches start
            vectorized_idx = self._compute_index_in_vectorized_data(axis_start_idx)
            patch_vectorized_indices.append(start_idx)
            patch_dims.append(axis_sample_lens)
        pass

    def sample_proj_mat(self, sample_inds):
        """
        Gets the projection matrix and it fits the transform to the samples of interest.

        Parameters
        ----------
        sample_inds : array of shape [n_samples]
            The data we are transforming.

        Returns
        -------
        proj_mat : {ndarray, sparse matrix} of shape (self.proj_dims, n_features)
            The generated sparse random matrix.
        proj_X : {ndarray, sparse matrix} of shape (n_samples, self.proj_dims)
            Projected input data matrix.

        Notes
        -----
        See `randMatTernary` in rerf.py for SPORF.

        See `randMat
        """

        if self.debug:
            return self.X[sample_inds, :], np.eye(self.n_features)

        # Edge case for fully dense matrix
        if self.density == 1:
            proj_mat = rng.binomial(1, 0.5, (self.n_features, self.proj_dims)) * 2 - 1

        # Sample random matrix and build it on that
        else:

            proj_mat = rng.rand(self.n_features, self.proj_dims)
            prob = 0.5 * self.density

            neg_inds = np.where(proj_mat <= prob)
            pos_inds = np.where(proj_mat >= 1 - prob)
            mid_inds = np.where((proj_mat > prob) & (proj_mat < 1 - prob))

            proj_mat[neg_inds] = -1
            proj_mat[pos_inds] = 1
            proj_mat[mid_inds] = 0

        proj_X = self.X[sample_inds, :] @ proj_mat

        return proj_X, proj_mat


class ManifoldObliqueTreeClassifier(ObliqueTreeClassifier):
    """"""

    def __init__(
        self,
        *,
        n_estimators=500,
        max_depth=None,
        min_samples_split=1,
        min_samples_leaf=1,
        min_impurity_decrease=0,
        min_impurity_split=0,
        bootstrap=True,
        feature_combinations=1.5,
        max_features=1,
        n_jobs=None,
        random_state=None,
        warm_start=False,
        verbose=0,
    ):

        super(ManifoldObliqueTreeClassifier, self).__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            feature_combinations=feature_combinations,
            max_features=max_features,
            random_state=random_state,
        )
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):

        # check random state - sklearn
        random_state = check_random_state(self.random_state)

        check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
        check_y_params = dict(ensure_2d=False, dtype=None)
        X, y = self._validate_data(
            X, y, validate_separately=(check_X_params, check_y_params)
        )
        if issparse(X):
            X.sort_indices()

            if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                raise ValueError(
                    "No support for np.int64 index based " "sparse matrices"
                )

        # Determine output settings
        n_samples, self.n_features_ = X.shape
        self.n_features_in_ = self.n_features_
        is_classification = is_classifier(self)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        # create the splitter
        splitter = MorfSplitter(
            X,
            y,
            self.max_features,
            self.feature_combinations,
            self.random_state,
            self.workers,
        )

        # create the Oblique tree
        self.tree = ObliqueTree(
            splitter,
            self.min_samples_split,
            self.min_samples_leaf,
            self.max_depth,
            self.min_impurity_split,
            self.min_impurity_decrease,
        )
        self.tree.build()
        return self