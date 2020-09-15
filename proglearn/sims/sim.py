import numpy as np
from sklearn.datasets import make_blobs


def _generate_2d_rotation(theta=0, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
    
    R = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    
    return R

def generate_gaussian_parity(n_samples, centers=None, class_label = None,
                             cluster_std=1.0, angle_params=None, random_state=None):
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
        centers = np.array(
            [(-.5,.5,), (.5,.5),
            (-.5,-.5), (.5,-.5)]
        )

    if class_label == None:
        class_label = [0,1,1,0]

    blob_num = len(class_label)

    #get the number of samples in each blob with equal probability
    samples_per_blob = np.random.multinomial(
        n_samples, 
        1/blob_num * np.ones(blob_num)
        )

    for blob in range(blob_num):
        x_, y_ = make_blobs(
            samples_per_blob[blob],
            n_features=2,
            centers=centers[blob].reshape(1,-1)
            )
        print(centers[blob].reshape(1,-1))
        if blob == 0:
            X = x_
            y = y_ + class_label[blob]
        else:
            X = np.concatenate((X,x_))
            y = np.concatenate(
                (y,
                y_+class_label[blob])
                )

    if angle_params != None:
        R = _generate_2d_rotation(angle_params)
        X = X @ R

    return X, y
