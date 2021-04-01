"""
Main Author: Will LeVine 
Corresponding Email: levinewill@icloud.com
"""
import keras
import numpy as np
import numpy.random as rng
from itertools import product
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .base import BaseTransformer

from split import BaseObliqueSplitter

class NeuralClassificationTransformer(BaseTransformer):
    """
    A class used to transform data from a category to a specialized representation.

    Parameters
    ----------
    network : object
        A neural network used in the classification transformer.

    euclidean_layer_idx : int
        An integer to represent the final layer of the transformer.

    optimizer : str or keras.optimizers instance
        An optimizer used when compiling the neural network.

    loss : str, default="categorical_crossentropy"
        A loss function used when compiling the neural network.

    pretrained : bool, default=False
        A boolean used to identify if the network is pretrained.

    compile_kwargs : dict, default={"metrics": ["acc"]}
        A dictionary containing metrics for judging network performance.

    fit_kwargs : dict, default={
                "epochs": 100,
                "callbacks": [keras.callbacks.EarlyStopping(patience=5, monitor="val_acc")],
                "verbose": False,
                "validation_split": 0.33,
            },
        A dictionary to hold epochs, callbacks, verbose, and validation split for the network.

    Attributes
    ----------
    encoder_ : object
        A Keras model with inputs and outputs based on the network attribute.
        Output layers are determined by the euclidean_layer_idx parameter.
    """

    def __init__(
        self,
        network,
        euclidean_layer_idx,
        optimizer,
        loss="categorical_crossentropy",
        pretrained=False,
        compile_kwargs={"metrics": ["acc"]},
        fit_kwargs={
            "epochs": 100,
            "callbacks": [keras.callbacks.EarlyStopping(patience=5, monitor="val_acc")],
            "verbose": False,
            "validation_split": 0.33,
        },
    ):
        self.network = keras.models.clone_model(network)
        self.encoder_ = keras.models.Model(
            inputs=self.network.inputs,
            outputs=self.network.layers[euclidean_layer_idx].output,
        )
        self.pretrained = pretrained
        self.optimizer = optimizer
        self.loss = loss
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs

    def fit(self, X, y):
        """
        Fits the transformer to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response data matrix).

        Returns
        -------
        self : NeuralClassificationTransformer
            The object itself.
        """
        check_X_y(X, y)
        _, y = np.unique(y, return_inverse=True)

        # more typechecking
        self.network.compile(
            loss=self.loss, optimizer=self.optimizer, **self.compile_kwargs
        )

        self.network.fit(X, keras.utils.to_categorical(y), **self.fit_kwargs)

        return self

    def transform(self, X):
        """
        Performs inference using the transformer.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        Returns
        -------
        X_transformed : ndarray
            The transformed input.

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        check_is_fitted(self)
        check_array(X)
        return self.encoder_.predict(X)


class TreeClassificationTransformer(BaseTransformer):
    """
    A class used to transform data from a category to a specialized representation.

    Parameters
    ----------
    kwargs : dict, default={}
        A dictionary to contain parameters of the tree.

    Attributes
    ----------
    transformer : sklearn.tree.DecisionTreeClassifier
        an internal sklearn DecisionTreeClassifier
    """

    def __init__(self, kwargs={}):
        self.kwargs = kwargs

    def fit(self, X, y):
        """
        Fits the transformer to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response data matrix).

        Returns
        -------
        self : TreeClassificationTransformer
            The object itself.
        """
        X, y = check_X_y(X, y)
        self.transformer_ = DecisionTreeClassifier(**self.kwargs).fit(X, y)
        return self

    def transform(self, X):
        """
        Performs inference using the transformer.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        Returns
        -------
        X_transformed : ndarray
            The transformed input.

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self.transformer_.apply(X)


class ObliqueTreeClassificationTransformer(BaseTransformer):
    """
    A class used to transform data from a category to a specialized representation.

    Parameters
    ----------
    kwargs : dict, default={}
        A dictionary to contain parameters of the tree.

    Attributes
    ----------
    transformer : ObliqueTreeClassifier
        an sklearn compliant oblique decision tree (SPORF)
    """

    def __init__(self, kwargs={}):
        self.kwargs = kwargs

    def fit(self, X, y):
        """
        Fits the transformer to data X with labels y.

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response data matrix).

        Returns
        -------
        self : TreeClassificationTransformer
            The object itself.
        """
        X, y = check_X_y(X, y)
        self.transformer_ = ObliqueTreeClassifier(**self.kwargs).fit(X, y)
        return self

    def transform(self, X):
        """
        Performs inference using the transformer.

        Parameters
        ----------
        X : ndarray
            Input data matrix.

        Returns
        -------
        X_transformed : ndarray
            The transformed input.

        Raises
        ------
        NotFittedError
            When the model is not fitted.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self.transformer_.apply(X)


"""
Authors: Parth Vora and Jay Mandavilli

Oblique Decision Tree (SPORF)
"""
# --------------------------------------------------------------------------
class SplitInfo:
    """
    A class used to store information about a certain split.

    Parameters
    ----------
    feature : int
        The feature which is used for the particular split.
    threshold : float
        The feature value which defines the split, if an example has a value less
        than this threshold for the feature of this split then it will go to the
        left child, otherwise it wil go the right child where these children are
        the children nodes of the node for which this split defines.
    proj_vec : array of shape [n_features]
        The vector of the sparse random projection matrix relevant for the split.
    left_impurity : float
        This is Gini impurity of left side of the split.
    left_idx : array of shape [left_n_samples]
        This is the indices of the nodes that are in the left side of this split.
    left_n_samples : int
        The number of samples in the left side of this split.
    right_impurity : float
        This is Gini impurity of right side of the split.
    right_idx : array of shape [right_n_samples]
        This is the indices of the nodes that are in the right side of this split.
    right_n_samples : int
        The number of samples in the right side of this split.
    no_split : bool
        A boolean specifying if there is a valid split or not. Here an invalid
        split means all of the samples would go to one side.
    improvement : float
        A metric to determine if the split improves the decision tree.
    """

    def __init__(
        self,
        feature,
        threshold,
        proj_vec,
        left_impurity,
        left_idx,
        left_n_samples,
        right_impurity,
        right_idx,
        right_n_samples,
        no_split,
        improvement,
    ):

        self.feature = feature
        self.threshold = threshold
        self.proj_vec = proj_vec
        self.left_impurity = left_impurity
        self.left_idx = left_idx
        self.left_n_samples = left_n_samples
        self.right_impurity = right_impurity
        self.right_idx = right_idx
        self.right_n_samples = right_n_samples
        self.no_split = no_split
        self.improvement = improvement


class ObliqueSplitter:
    """
    A class used to represent an oblique splitter, where splits are done on
    the linear combination of the features.

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
    sample_proj_mat(sample_inds)
        This gets the projection matrix and it fits the transform to the samples of interest.
    leaf_label_proba(idx)
        This calculates the label and the probability for that label for a particular leaf
        node.
    score(y_sort, t)
        Finds the Gini impurity for a split.
    _score(self, proj_X, y_sample, i, j)
        Handles array indexing before calculating Gini impurity.
    impurity(idx)
        Finds the impurity for a certain set of samples.
    split(sample_inds)
        Determines the best possible split for the given set of samples.
    """

    def __init__(self, X, y, max_features, feature_combinations, random_state, n_jobs):

        self.X = np.array(X, dtype=np.float64)

        # y must be 1D
        self.y = np.array(y, dtype=np.float64).squeeze()

        classes = np.array(np.unique(y), dtype=int)
        self.n_classes = len(classes)
        self.class_indices = {j:i for i, j in enumerate(classes)}

        self.indices = np.indices(self.y.shape)[0]

        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]

        self.random_state = random_state
        rng.seed(random_state)

        # Compute root impurity
        unique, count = np.unique(y, return_counts=True)
        count = count / len(y)
        self.root_impurity = 1 - np.sum(np.power(count, 2))

        # proj_dims = d = mtry
        self.proj_dims = max(int(max_features * self.n_features), 1)
       
        # feature_combinations = mtry_mult
        # In processingNodeBin.h, mtryDensity = int(mtry * mtry_mult)
        self.n_non_zeros = max(int(self.proj_dims * feature_combinations), 1)

        # Base oblique splitter in cython
        self.BOS = BaseObliqueSplitter()

        # Temporary debugging parameter, turns off oblique splits
        self.debug = False

        self.n_jobs = n_jobs

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
        """
        
        if self.debug:
            return self.X[sample_inds, :], np.eye(self.n_features)

        # This is the way its done in the C++
        proj_mat = np.zeros((self.n_features, self.proj_dims))
        rand_feat = rng.randint(self.n_features, size=self.n_non_zeros)
        rand_dim = rng.randint(self.proj_dims, size=self.n_non_zeros)
        weights = [1 if rng.rand() > 0.5 else -1 for i in range(self.n_non_zeros)]
        proj_mat[rand_feat, rand_dim] = weights

        # Need to remove zero vectors from projmat
        proj_mat = proj_mat[:, np.unique(rand_dim)]
        
        proj_X = self.X[sample_inds, :] @ proj_mat
        return proj_X, proj_mat

    def leaf_label_proba(self, idx):
        """
        Finds the most common label and probability of this label from the samples at
        the leaf node for which this is used on.

        Parameters
        ----------
        idx : array of shape [n_samples]
            The indices of the samples that are at the leaf node for which the label
            and probability need to be found.

        Returns
        -------
        label : int
            The label for any sample that is predicted to be at this node.
        proba : float
            The probability of the predicted sample to have this node's label.
        """

        samples = self.y[idx]
        n = len(samples)
        labels, count = np.unique(samples, return_counts=True)
       
        proba = np.zeros(self.n_classes)
        for i, l in enumerate(labels):
            class_idx = self.class_indices[l]
            proba[class_idx] = count[i] / n

        most = np.argmax(count)
        label = labels[most]
        #max_proba = count[most] / n

        return label, proba

    # Finds the best split
    def split(self, sample_inds):
        """
        Finds the optimal split for a set of samples.

        Parameters
        ----------
        sample_inds : array of shape [n_samples]
            The indices of the nodes in the set for which the best split is found.

        Returns
        -------
        split_info : SplitInfo
            Class holding information about the split.
        """
        # ensure that sample indices are 1D
        sample_inds = sample_inds.squeeze()

        if not self.y.squeeze().ndim == 1:
            raise RuntimeError('Does not support multivariate output yet.')

        # Project the data
        proj_X, proj_mat = self.sample_proj_mat(sample_inds)
        y_sample = self.y[sample_inds]
        n_samples = len(sample_inds)

        # Assign types to everything
        # TODO: assign types when making the splitter class. This is silly.
        proj_X = np.array(proj_X, dtype=np.float64)
        y_sample = np.array(y_sample, dtype=np.float64)
        sample_inds = np.array(sample_inds, dtype=np.intc)
        # print(proj_X.shape, y_sample.shape, sample_inds.shape)

        # Call cython splitter 
        (feature, 
        threshold, 
        left_impurity, 
        left_idx, 
        right_impurity, 
        right_idx, 
        improvement) = self.BOS.best_split(proj_X, y_sample, sample_inds)

        # TODO: These are coming out as memory views. Fix this.
        left_idx = np.asarray(left_idx)
        right_idx = np.asarray(right_idx)
    
        left_n_samples = len(left_idx)
        right_n_samples = len(right_idx)

        no_split = left_n_samples == 0 or right_n_samples == 0

        proj_vec = proj_mat[:, feature]

        split_info = SplitInfo(
            feature,
            threshold,
            proj_vec,
            left_impurity,
            left_idx,
            left_n_samples,
            right_impurity,
            right_idx,
            right_n_samples,
            no_split,
            improvement,
        )

        return split_info


# --------------------------------------------------------------------------


class Node:
    """
    A class used to represent an oblique node.

    Parameters
    ----------
    None

    Methods
    -------
    None
    """

    def __init__(self):
        self.node_id = None
        self.is_leaf = None
        self.parent = None
        self.left_child = None
        self.right_child = None

        self.feature = None
        self.threshold = None
        self.impurity = None
        self.n_samples = None

        self.proj_vec = None
        self.label = None
        self.proba = None


class StackRecord:
    """
    A class used to keep track of a node's parent and other information about the node and its split.

    Parameters
    ----------
    parent : int
        The index of the parent node.
    depth : int
        The depth at which this node is.
    is_left : bool
        Represents if the node is a left child or not.
    impurity : float
        This is Gini impurity of this node.
    sample_idx : array of shape [n_samples]
        This is the indices of the nodes that are in this node.
    n_samples : int
        The number of samples in this node.

    Methods
    -------
    None
    """

    def __init__(self, parent, depth, is_left, impurity, sample_idx, n_samples):

        self.parent = parent
        self.depth = depth
        self.is_left = is_left
        self.impurity = impurity
        self.sample_idx = sample_idx
        self.n_samples = n_samples


class ObliqueTree:
    """
    A class used to represent a tree with oblique splits.

    Parameters
    ----------
    splitter : class
        The type of splitter for this tree, should be an ObliqueSplitter.
    min_samples_split : int
        Minimum number of samples possible at a node.
    min_samples_leaf : int
        Minimum number of samples possible at a leaf.
    max_depth : int
        Maximum depth allowed for the tree.
    min_impurity_split : float
        Minimum Gini impurity value that must be achieved for a split to occur on the node.
    min_impurity_decrease : float
        Minimum amount Gini impurity value must decrease by for a split to be valid.

    Methods
    -------
    add_node(parent, is_left, impurity, n_samples, is_leaf, feature, threshold, proj_mat, label, proba)
        Adds a node to the existing tree
    build()
        This is what is initially called on to completely build the oblique tree.
    predict(X)
        Finds the final node for each input sample as it passes through the decision tree.
    """

    def __init__(
        self,
        splitter,
        min_samples_split,
        min_samples_leaf,
        max_depth,
        min_impurity_split,
        min_impurity_decrease,
    ):

        # Tree parameters
        self.depth = 0
        self.node_count = 0
        self.nodes = []

        # Build parameters
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_split = min_impurity_split
        self.min_impurity_decrease = min_impurity_decrease

    def add_node(
        self,
        parent,
        is_left,
        impurity,
        n_samples,
        is_leaf,
        feature,
        threshold,
        proj_vec,
        label,
        proba,
    ):
        """
        Adds a node to the existing oblique tree.

        Parameters
        ----------
        parent : int
            The index of the parent node for the new node being added.
        is_left : bool
            Determines if this new node being added is a left or right child.
        impurity : float
            Impurity of this new node.
        n_samples : int
            Number of samples at this new node.
        is_leaf : bool
            Determines if this new node is a leaf of the tree or an internal node.
        feature : int
            Index of feature on which the split occurs at this node.
        threshold : float
            The threshold feature value for this node determining if a sample will go
            to this node's left of right child. If a sample has a value less than the
            threshold (for the feature of this node) it will go to the left childe,
            otherwise it will go the right child.
        proj_vec : {ndarray, sparse matrix} of shape (n_features)
            Projection vector for this new node.
        label : int
            The label a sample will be given if it is predicted to be at this node.
        proba : float
            The probability a predicted sample has of being the node's label.

        Returns
        -------
        node_id : int
            Index of the new node just added.
        """

        node = Node()
        node.node_id = self.node_count
        node.impurity = impurity
        node.n_samples = n_samples

        # If not the root node, set parents
        if self.node_count > 0:
            node.parent = parent
            if is_left:
                self.nodes[parent].left_child = node.node_id
            else:
                self.nodes[parent].right_child = node.node_id

        # Set node parameters
        if is_leaf:
            node.is_leaf = True
            node.label = label
            node.proba = proba
        else:
            node.is_leaf = False
            node.feature = feature
            node.threshold = threshold
            node.proj_vec = proj_vec

        self.node_count += 1
        self.nodes.append(node)

        return node.node_id

    def build(self):
        """
        Builds the oblique tree.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Initialize, add root node
        stack = []
        root = StackRecord(
            0,
            1,
            False,
            self.splitter.root_impurity,
            self.splitter.indices,
            self.splitter.n_samples,
        )
        stack.append(root)

        # Build tree
        while len(stack) > 0:

            # Pop a record off the stack
            cur = stack.pop()

            # Evaluate if it is a leaf
            is_leaf = (
                cur.depth >= self.max_depth
                or cur.n_samples < self.min_samples_split
                or cur.n_samples < 2 * self.min_samples_leaf
                or cur.impurity <= self.min_impurity_split
            )

            # Split if not
            if not is_leaf:
                split = self.splitter.split(cur.sample_idx)

                is_leaf = (
                    is_leaf
                    or split.no_split
                    or split.improvement <= self.min_impurity_decrease
                )

            # Add the node to the tree
            if is_leaf:

                label, proba = self.splitter.leaf_label_proba(cur.sample_idx)

                node_id = self.add_node(
                    cur.parent,
                    cur.is_left,
                    cur.impurity,
                    cur.n_samples,
                    is_leaf,
                    None,
                    None,
                    None,
                    label,
                    proba,
                )

            else:
                node_id = self.add_node(
                    cur.parent,
                    cur.is_left,
                    cur.impurity,
                    cur.n_samples,
                    is_leaf,
                    split.feature,
                    split.threshold,
                    split.proj_vec,
                    None,
                    None,
                )

            # Push the right and left children to the stack if applicable
            if not is_leaf:

                right_child = StackRecord(
                    node_id,
                    cur.depth + 1,
                    False,
                    split.right_impurity,
                    split.right_idx,
                    split.right_n_samples,
                )
                stack.append(right_child)

                left_child = StackRecord(
                    node_id,
                    cur.depth + 1,
                    True,
                    split.left_impurity,
                    split.left_idx,
                    split.left_n_samples,
                )
                stack.append(left_child)

            if cur.depth > self.depth:
                self.depth = cur.depth

    def predict(self, X):
        """
        Predicts final nodes of samples given.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input array for which predictions are made.

        Returns
        -------
        predictions : array of shape [n_samples]
            Array of the final node index for each input prediction sample.
        """

        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            cur = self.nodes[0]
            while cur is not None and not cur.is_leaf:
               
                proj_X = X[i] @ cur.proj_vec

                if proj_X < cur.threshold:
                    id = cur.left_child
                    cur = self.nodes[id]
                else:
                    id = cur.right_child
                    cur = self.nodes[id]

            predictions[i] = cur.node_id

        return predictions


# --------------------------------------------------------------------------

""" Class for Oblique Tree """


class ObliqueTreeClassifier(BaseEstimator):
    """
    A class used to represent a classifier that uses an oblique decision tree.

    Parameters
    ----------
    max_depth : int
        Maximum depth allowed for oblique tree.
    min_samples_split : int
        Minimum number of samples possible at a node.
    min_samples_leaf : int
        Minimum number of samples possible at a leaf.
    random_state : int
        Maximum depth allowed for the tree.
    min_impurity_decrease : float
        Minimum amount Gini impurity value must decrease by for a split to be valid.
    min_impurity_split : float
        Minimum Gini impurity value that must be achieved for a split to occur on the node.
    feature_combinations : float
        The feature combinations to use for the oblique split.
    max_features : float
        Output dimension = max_features * dimension
    n_jobs : int, optional (default: -1)
        The number of cores to parallelize the calculation of Gini impurity.
        Supply -1 to use all cores available to the Process.

    Methods
    -------
    fit(X,y)
        Fits the oblique tree to the training samples.
    apply(X)
        Calls on the predict function from the oblique tree for the test samples.
    predict(X)
        Gets the prediction labels for the test samples.
    predict_proba(X)
        Gets the probability of the prediction labels for the test samples.
    predict_log_proba(X)
        Gets the log of the probability of the prediction labels for the test samples.
    """

    def __init__(
        self,
        *,
        max_depth=np.inf,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        min_impurity_decrease=0,
        min_impurity_split=0,
        feature_combinations=1.5,
        max_features=1,
        n_jobs=1
    ):

        # RF parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

        # OF parameters
        # Feature combinations = L
        self.feature_combinations = feature_combinations

        # Max features
        self.max_features = max_features

        self.n_jobs=n_jobs

        self.n_classes=None

    def fit(self, X, y):
        """
        Predicts final nodes of samples given.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The training samples.
        y : array of shape [n_samples]
            Labels for the training samples.

        Returns
        -------
        ObliqueTreeClassifier
            The fit classifier.
        """

        splitter = ObliqueSplitter(
            X, y, self.max_features, self.feature_combinations, self.random_state, self.n_jobs
        )
        self.n_classes = splitter.n_classes

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

    def apply(self, X):
        """
        Gets predictions form the oblique tree for the test samples.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The testing samples.

        Returns
        -------
        pred_nodes : array of shape[n_samples]
            The indices for each test sample's final node in the oblique tree.
        """

        pred_nodes = self.tree.predict(X).astype(int)
        return pred_nodes

    def predict(self, X):
        """
        Determines final label predictions for each sample in the test data.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The testing samples.

        Returns
        -------
        preds : array of shape[n_samples]
            The predictions (labels) for each testing sample.
        """

        X = np.array(X, dtype=np.float64)

        preds = np.zeros(X.shape[0])
        pred_nodes = self.apply(X)
        for k in range(len(pred_nodes)):
            id = pred_nodes[k]
            preds[k] = self.tree.nodes[id].label

        return preds

    def predict_proba(self, X):
        """
        Determines probabilities of the final label predictions for each sample in the test data.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The testing samples.

        Returns
        -------
        preds : array of shape[n_samples]
            The probabilities of the predictions (labels) for each testing sample.
        """

        X = np.array(X, dtype=np.float64)

        preds = np.zeros((X.shape[0], self.n_classes))
        pred_nodes = self.apply(X)
        for k in range(len(preds)):
            id = pred_nodes[k]
            preds[k] = self.tree.nodes[id].proba

        return preds

    def predict_log_proba(self, X):
        """
        Determines log of the probabilities of the final label predictions for each sample in the test data.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The testing samples.

        Returns
        -------
        preds : array of shape[n_samples]
            The log of the probabilities of the predictions (labels) for each testing sample.
        """

        proba = self.predict_proba(X)
        return np.log(proba)


