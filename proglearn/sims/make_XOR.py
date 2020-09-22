import itertools
import numpy as np
from sklearn import datasets
from sklearn.utils import check_random_state

def make_XOR(n_samples=100, cluster_center=[0,0], cluster_std=0.25, 
             dist_from_center=0.5, N_XOR=False, theta_rotation=0,
             random_state=None):
    """
    Generate 2-dimensional Gaussian XOR distribution.
    (Classic XOR problem but each point is the 
     center of a Gaussian blob distribution)
    
    Parameters
    ----------
    n_samples : int, optional (default=100)
        If int, it is the total number of points equally divided among
        the four clusters.
        
    cluster_center : array of shape [2,], optional (default=[0,0])
        The x1 and x2 coordinates of the center of the four clusters.

    cluster_std : float, optional (default=0.25)
        The standard deviation of the clusters.
    
    dist_from_center : float, optional (default=0.5)
        X value distance of each cluster to the center of the four clusters.
        
    N_XOR : boolean, optional (default=False)
        Change to Gaussian N_XOR distribution (inverts the class labels).
        
    theta_rotation : float, optional (default=0)
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
    
    #variable setup
    seed = random_state
    dist = dist_from_center
    std = cluster_std
    n = int(n_samples/4)
    
    cluster_centers = np.array(list(itertools.product([dist, -dist], repeat=2)))
    cluster_centers = cluster_center - cluster_centers
    n_per_cluster = np.full(shape=2, fill_value=n)

    #make blobs
    X1,_ = datasets.make_blobs(n_samples=n_per_cluster, n_features=2, 
                               centers=cluster_centers[[0,3], :], 
                               cluster_std=std, random_state=seed)
    X2,_ = datasets.make_blobs(n_samples=n_per_cluster, n_features=2, 
                               centers=cluster_centers[[1,2], :], 
                               cluster_std=std, random_state=seed)
    
    #assign classe
    if N_XOR:
        y1, y2 = np.zeros(n*2), np.ones(n*2)
    else:
        y1, y2 = np.ones(n*2), np.zeros(n*2)

    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, y2))

    #rotation
    c, s = np.cos(theta_rotation), np.sin(theta_rotation)
    R = np.array([[c, -s], [s, c]])
    X = (R @ X.T).T
    
    return X,y
