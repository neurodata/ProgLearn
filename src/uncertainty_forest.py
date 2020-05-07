#Model
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#Infrastructure
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import NotFittedError

#Data Handling
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)
from sklearn.utils.multiclass import check_classification_targets

#Utils
from joblib import Parallel, delayed
import numpy as np

from tqdm import tqdm

def _finite_sample_correction(posteriors, num_points_in_partition, num_classes):
    '''
    encourage posteriors to approach uniform when there is low data
    '''
    correction_constant = 1 / (num_classes * num_points_in_partition)

    zero_posterior_idxs = np.where(posteriors == 0)[0]

    c = len(zero_posterior_idxs) / (num_classes * num_points_in_partition)
    posteriors *= (1 - c)
    posteriors[zero_posterior_idxs] = correction_constant
    return posteriors

class UncertaintyForest(BaseEstimator, ClassifierMixin):
    '''
    based off of https://arxiv.org/pdf/1907.00325.pdf
    '''
    def __init__(
        self,
        max_depth=30,
        min_samples_leaf=10,
        max_samples = 0.5,
        max_features_tree = "auto",
        n_estimators=3500,
        bootstrap=False,
        parallel=True,
        calibration_split = .33
    ):

        #Tree parameters.
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features_tree = max_features_tree

        #Bag parameters
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples

        #Model parameters.
        self.parallel = parallel
        self.calibration_split = calibration_split
        self.fitted = False

    def _check_fit(self):
        '''
        raise a NotFittedError if the model isn't fit
        '''
        if not self.fitted:
                msg = (
                        "This %(name)s instance is not fitted yet. Call 'fit' with "
                        "appropriate arguments before using this estimator."
                )
                raise NotFittedError(msg % {"name": type(self).__name__})

    
        

    def _get_nodes_and_tree_depth(self, tree, X):
        '''
        given a tree, return its depth and the leaf nodes of the input X
        '''
        binary_decision_paths = tree.decision_path(X).toarray()
        tree_depth = np.shape(binary_decision_paths)[1]
        base_2_powers = 2 ** np.arange(0, tree_depth, 1)
        decision_paths = binary_decision_paths * base_2_powers
        nodes = np.sum(decision_paths, axis = 1)
        return nodes, tree_depth
    
    def transform(self, X):
        '''
        get the estimated posteriors across trees
        '''
        print("Transforming Points")
        self._check_fit()
        X = check_array(X)
        
        #for aggreg
        node_ids = []

        def worker(tree_idx, tree):
            #get the nodes of X
            nodes, _= self._get_nodes_and_tree_depth(tree, X)
            return nodes

            

        if self.parallel:
            return np.array(
                    Parallel(n_jobs=-1)(
                            delayed(worker)(tree_idx, tree) for tree_idx, tree in enumerate(self.ensemble.estimators_)
                    )
            )
                    
        else:
            return np.array(
                    [worker(tree_idx, tree) for tree_idx, tree in enumerate(self.ensemble.estimators_)]
                    )
        
    def get_transformer(self):
        return lambda X : self.transform(X)
        
    def vote(self, nodes_across_trees):
        print("Voting On Points")
        return self.voter.predict(nodes_across_trees)
        
    def get_voter(self):
        return self.voter
        
                        
    def fit(self, X, y):

        #format X and y
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)

        #split into train and cal
        X_train, X_cal, y_train, y_cal = train_test_split(X, y)
        
        
        #define the ensemble
        self.ensemble = BaggingClassifier(
            DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features_tree,
                splitter="random"
            ),
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            bootstrap=self.bootstrap,
            n_jobs = -1,
            verbose = 1
        )
        
        print("Fitting Transformer")
        #fit the ensemble
        self.ensemble.fit(X_train, y_train)
        self.fitted = True
        
        class Voter(BaseEstimator):
            def __init__(self, n_estimators, classes, parallel = True):
                self.n_estimators = n_estimators
                self.classes_ = classes
                self.parallel = parallel
            
            def fit(self, cal_nodes_across_trees, y_cal):
                self.tree_idx_to_node_ids_to_posterior_map = {}

                def worker(tree_idx):
                    cal_nodes = cal_nodes_across_trees[tree_idx]
                    #create a map from the unique node ids to their classwise posteriors
                    node_ids_to_posterior_map = {}

                    #fill in the posteriors 
                    for node_id in np.unique(cal_nodes):
                        cal_idxs_of_node_id = np.where(cal_nodes == node_id)[0]
                        cal_ys_of_node = y_cal[cal_idxs_of_node_id]
                        class_counts = [len(np.where(cal_ys_of_node == y)[0]) for y in np.unique(y_train) ]
                        posteriors = np.nan_to_num(np.array(class_counts) / np.sum(class_counts))

                        #finite sample correction
                        posteriors_corrected = _finite_sample_correction(posteriors, len(cal_idxs_of_node_id), len(self.classes_))
                        node_ids_to_posterior_map[node_id] = posteriors_corrected

                    #add the node_ids_to_posterior_map to the overall tree_idx map 
                    self.tree_idx_to_node_ids_to_posterior_map[tree_idx] = node_ids_to_posterior_map
                    
                for tree_idx in tqdm(range(self.n_estimators)):
                        worker(tree_idx)
                return self
                        
                        
            def predict_proba(self, nodes_across_trees):
                def worker(tree_idx):
                    #get the node_ids_to_posterior_map for this tree
                    node_ids_to_posterior_map = self.tree_idx_to_node_ids_to_posterior_map[tree_idx]

                    #get the nodes of X
                    nodes = nodes_across_trees[tree_idx]

                    posteriors = []
                    node_ids = node_ids_to_posterior_map.keys()

                    #loop over nodes of X
                    for node in nodes:
                        #if we've seen this node before, simply get the posterior
                        if node in node_ids:
                            posteriors.append(node_ids_to_posterior_map[node])
                        #if we haven't seen this node before, simply use the uniform posterior 
                        else:
                            posteriors.append(np.ones((len(np.unique(self.classes_)))) / len(self.classes_))
                    return posteriors

                if self.parallel:
                    return np.mean(
                            Parallel(n_jobs=-1)(
                                    delayed(worker)(tree_idx) for tree_idx in range(self.n_estimators)
                            ), axis = 0
                    )

                else:
                    return np.mean(
                            [worker(tree_idx) for tree_idx in range(self.n_estimators)],
                            axis = 0)
                
        #get the nodes of the calibration set
        cal_nodes_across_trees = self.transform(X_cal) 
        print("Fitting Voter")
        self.voter = Voter(n_estimators = len(self.ensemble.estimators_), classes = self.classes_, parallel = self.parallel)
        self.voter.fit(cal_nodes_across_trees, y_cal)
        

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=-1)]

    def predict_proba(self, X):
        return self.voter.predict_proba(self.transform(X))

    def estimate_conditional_entropy(self, X):
        '''
        as described in Algorithm 1
        '''
        posteriors_across_trees = self._get_posteriors_across_trees(X)
        conditional_entropy_across_trees = np.sum(-posteriors_across_trees * np.log(posteriors_across_trees), axis = -1)

        return np.mean(conditional_entropy_across_trees, axis = 0)