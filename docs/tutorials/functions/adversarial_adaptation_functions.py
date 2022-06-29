import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

from math import log2, ceil
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

from pycpd import RigidRegistration
from pycpd import AffineRegistration
from pycpd import DeformableRegistration

from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter
from proglearn.sims import generate_gaussian_parity

import ot
# import SimpleITK as sitk  # need to build from source in order to install
from graspologic.align import SeedlessProcrustes
import random
from scipy.spatial.distance import cdist
import warnings

## TRANSFORMATION FUNCTIONS ##--------------------------------------------------------------------------------------

#Function to rotate distribution
def rotate(X, theta=0, dim=[0,1]):
    #dimensions to rotate
    Z = X[:, dim]
    
    #rotation
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    Z = Z @ R
    
    #recombine
    X[:, dim] = Z
    
    return X


#Function to shear in the X direction
def shearX(X, s=0, dim=[0,1]):
    #dimensions to shear
    Z = X[:, dim]
    
    #shear
    R = np.array([[1, 0], [s, 1]])
    Z = Z @ R
    
    #recombine
    X[:, dim] = Z
    
    return X


#Function to shear in the Y direction
def shearY(X, s=0, dim=[0,1]):
    #dimensions to shear
    Z = X[:, dim]
    
    #shear
    R = np.array([[1, s], [0, 1]])
    Z = Z @ R
    
    #recombine
    X[:, dim] = Z
    
    return X


#Function to double shear in the X direction
def double_shearX(X, y, ss=(0,0) , dim=[0,1]):
    #dimensions to shear
    Z = X[:, dim]
    i,j = dim
    t,b = ss
    
    Z_top = Z[Z[:,j] >= 0]
    Z_bot = Z[Z[:,j] < 0]
    c_top = y[Z[:,j] >= 0]
    c_bot = y[Z[:,j] < 0]
    
    #shear
    R_top = np.array([[1, 0], [t, 1]])
    R_bot = np.array([[1, 0], [b, 1]])
    Z_top = Z_top @ R_top
    Z_bot = Z_bot @ R_bot
    
    #recombine
    Z = np.concatenate((Z_top, Z_bot))
    y = np.concatenate((c_top, c_bot))
    X[:, dim] = Z
    
    return X, y


#Function to divergently translate in the X direction
def div_translateX(X, y, t=0, dim=[0,1]):
    #dimensions to translate
    Z = X[:, dim]
    i,j = dim
    
    Z_top = Z[Z[:,j] >= 0]
    Z_bot = Z[Z[:,j] < 0]
    c_top = y[Z[:,j] >= 0]
    c_bot = y[Z[:,j] < 0]
    
    #stranslate
    Z_top[:, i] = Z_top[:, i] + t
    Z_bot[:, i] = Z_bot[:, i] - t
    
    #recombine
    Z = np.concatenate((Z_top, Z_bot))
    y = np.concatenate((c_top, c_bot))
    X[:, dim] = Z
    
    return X, y

## ADAPTATION ALGORITHMS ##--------------------------------------------------------------------------------------

def nearest_neighbor(src, dst, y_src, y_dst, class_aware=True):
    #assert src.shape == dst.shape

    distances = np.zeros(y_src.shape)
    indices = np.zeros(y_src.shape, dtype=int)

    if class_aware:
        class1_src = np.where(y_src == 1)[0]
        class0_src = np.where(y_src == 0)[0]
        class1_dst = np.where(y_dst == 1)[0]
        class0_dst = np.where(y_dst == 0)[0]

        neigh_1 = NearestNeighbors(n_neighbors=1)
        neigh_1.fit(dst[class1_dst])
        distances_1, indices_1 = neigh_1.kneighbors(
            src[class1_src], return_distance=True
        )

        neigh_2 = NearestNeighbors(n_neighbors=1)
        neigh_2.fit(dst[class0_dst])
        distances_2, indices_2 = neigh_2.kneighbors(
            src[class0_src], return_distance=True
        )

        closest_class1 = class1_src[indices_1]
        closest_class0 = class0_src[indices_2]

        count = 0
        for i in class1_src:
            distances[i] = distances_1[count]
            indices[i] = closest_class1[count]
            count = count + 1

        count = 0
        for i in class0_src:
            distances[i] = distances_2[count]
            indices[i] = closest_class0[count]
            count = count + 1

    else:
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)

    return distances.ravel(), indices.ravel()


def best_fit_transform(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def icp(A, B, y_src, y_dst, init_pose=None, max_iterations=500, tolerance=1e-26):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    imbalance = []

    class1_src = np.where(y_src == 1)[0]
    class0_src = np.where(y_src == 0)[0]
    class1_dst = np.where(y_dst == 1)[0]
    class0_dst = np.where(y_dst == 0)[0]

    imbalance.append(len(class1_src))
    imbalance.append(len(class0_src))
    imbalance.append(len(class1_dst))
    imbalance.append(len(class0_dst))

    mi = min(imbalance)

    X_1 = src[:, class1_src[0:mi]]
    X_2 = src[:, class0_src[0:mi]]

    src_subsample = np.concatenate((X_1, X_2), 1)
    y_src_sub = np.concatenate((np.ones(mi), np.zeros(mi)))

    X_1 = dst[:, class1_dst[0:mi]]
    X_2 = dst[:, class0_dst[0:mi]]
    dst_subsample = np.concatenate((X_1, X_2), 1)
    y_dst_sub = np.concatenate((np.ones(mi), np.zeros(mi)))

    for i in range(max_iterations):

        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(
            src_subsample[:m, :].T, dst_subsample[:m, :].T, y_src_sub, y_dst_sub
        )
        # distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T, y_src, y_dst)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(
            src_subsample[:m, :].T, dst_subsample[:m, indices].T
        )
        # T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src_subsample = np.dot(T, src_subsample)
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    # T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, src, i

def cpd_reg(template, target, max_iter=200):    
    registration = AffineRegistration(X=target, Y=template, max_iterations=max_iter)
    deformed_template = registration.register(template)
    
    return deformed_template[0]

def nlr_reg(template, target, max_iter=200, alpha=0.001, beta=2):    
    registration = DeformableRegistration(X=target, Y=template, alpha=alpha, beta=beta, max_iterations=max_iter, tolerance=1e-8)
    deformed_template = registration.register(template)
    
    return deformed_template[0]


# register posteriors using Optimal Transport
def exp_reg_pos_OT(angle, transform, n_trees=10, n_samples_source=200):
    [train_x1, train_x2_rot, test_x1, test_y1, test_x2_rot, test_y2], l2f, uf =\
        get_data(angle, transform, n_samples_source, n_trees)   
    reg = [1.0, 1.0, 1.0]
    if transform == 0:
        # use Sinkhorn Transport
        OT = ot.da.SinkhornTransport(reg_e=reg[transform], tol=10e-15)
        OT.fit(Xs=train_x1, Xt=train_x2_rot)
        test_x1_trans = OT.transform(Xs=test_x1)
        test_x2_trans = OT.inverse_transform(Xt=test_x2_rot)
        w = 1  # weight ratio for weighted average
    else:
        # use Seedless Procrustes
        SP = SeedlessProcrustes(optimal_transport_lambda=reg[transform], optimal_transport_eps=10e-15,
                               optimal_transport_num_reps=10)
        SP.fit(train_x1, train_x2_rot)
        test_x1_trans = SP.transform(test_x1)
        SP = SeedlessProcrustes(optimal_transport_lambda=reg[transform], optimal_transport_eps=10e-15,
                               optimal_transport_num_reps=10)
        SP.fit(train_x2_rot, train_x1)
        test_x2_trans = SP.transform(test_x2_rot)
        w = 2  # weight ratio for weighted average

    errors = np.zeros(6)
    # UF
    uf_task1 = uf.predict(test_x1, transformer_ids=[0], task_id=0)
    uf_task2 = uf.predict(test_x2_rot, transformer_ids=[1], task_id=1)
    errors[0] = 1 - np.mean(uf_task1 == test_y1)
    errors[1] = 1 - np.mean(uf_task2 == test_y2)

    # L2F
    l2f_task1 = l2f.predict(test_x1, task_id=0)
    l2f_task2 = l2f.predict(test_x2_rot, task_id=1)
    errors[2] = 1 - np.mean(l2f_task1 == test_y1)
    errors[3] = 1 - np.mean(l2f_task2 == test_y2)

    # OT
    l2f_task1_pos = generate_posteriors(test_x1, 0, l2f, [0,1])[0]
    l2f_task2_pos = generate_posteriors(test_x2_rot, 1, l2f, [0,1])[1]
    l2f_task1_pos_trans = generate_posteriors(test_x1_trans, 1, l2f, [0,1])[1]
    l2f_task2_pos_trans = generate_posteriors(test_x2_trans, 0, l2f, [0,1])[0]

    OT_task1 = np.average([l2f_task1_pos, l2f_task1_pos_trans], axis=0, weights=[w,1])
    OT_task2 = np.average([l2f_task2_pos, l2f_task2_pos_trans], axis=0, weights=[w,1])

    errors[4] = 1 - np.mean(np.argmax(OT_task1, axis=1) == test_y1)
    errors[5] = 1 - np.mean(np.argmax(OT_task2, axis=1) == test_y2)

    return errors


# generate a test grid in a ciclr
def to_grid_in_cir(train_task1, train_task2, h = 0.075):
    # compute largest distance to origin
    dists = cdist(np.vstack((train_task1, train_task2)), np.array([[0,0]]))
    radius = np.max(dists)
    x, y = np.meshgrid(np.arange(-radius, radius, h), np.arange(-radius, radius, h))
    r = x**2 + y**2  # centered at (0,0)
    inside = r <= radius**2 + 0.1
    x_cir = x[inside]
    y_cir = y[inside]
    return x, y, inside, np.array([x_cir, y_cir]).T, radius


# reshape a vector of posteriors to match the shape of the test_grid
def reshape_posteriors(inside, task):
    inside_ravel = inside.ravel()
    preds = np.zeros(inside_ravel.shape)
    j = 0
    for i in range(len(inside_ravel)):
        if inside_ravel[i]:
            preds[i] = task[j]
            j += 1
    preds = preds.reshape(inside.shape)
    
    return preds


# transform test points for rigid transformation
def trans_pts_rigid(p, test):
    theta = float(p[0])  # the rotation parameter learnt from the morphing/transformation
    Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    test_trans = np.matmul(test, Q)
    test_trans[:,0] + float(p[1])
    test_trans[:,1] + float(p[2])
    
    return test_trans


# predict for test data
def pred_on_task(l2f, trans_id, test_grid, test, n_trees, taskpred_deformed, inside):
    leaf_pos = []
    for n in range(n_trees):
        # get each tree
        transformer_ = l2f.transformer_id_to_transformers[trans_id][n].transformer_
        # record the leaf node each point on test_grid/test (data) is mapped to
        leaf_pred_grid = transformer_.apply(test_grid)
        leaf_pred_test = transformer_.apply(test)
        leaf_pos_ = np.zeros(len(test))
        for i in range(len(test)):
            idx = leaf_pred_grid == leaf_pred_test[i]
            leaf_pos_[i] = np.nanmean(taskpred_deformed[inside][idx])
        leaf_pos.append(leaf_pos_) 
    return np.vstack((np.nanmean(leaf_pos, axis=0), 1-np.nanmean(leaf_pos, axis=0))).T


# define a parameter map to run SimpleElastix
def run_elastix(task1pred1_reshaped, task2pred2_reshaped, res, ite, scale, trans):
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(task1pred1_reshaped))
    elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(task2pred2_reshaped))
    elastixImageFilter.LogToFileOn()
    elastixImageFilter.LogToConsoleOff()
    ParamMap = sitk.GetDefaultParameterMap('affine')    
    ParamMap['AutomaticTransformInitializationMethod'] = ['GeometricalCenter']
    ParamMap['Metric'] = ['AdvancedNormalizedCorrelation']
    ParamMap['MaximumNumberOfIterations'] = [ite]
    ParamMap['Transform'] = [trans]
    ParamMap['AutomaticTransformInitialization'] = ['true']
    ParamMap['NumberOfResolutions'] = [res]
    ParamMap['MaximumStepLength'] = ['0.1']
    ParamMap['ImageSampler'] = ['Random']
    ParamMap['SP_alpha'] = ['0.6']
    ParamMap['SP_A'] = ['50']
    ParamMap['NewSamplesEveryIteration'] = ['true']
    ParamMap['FixedImagePyramid'] = ['FixedRecursiveImagePyramid']
    ParamMap["MovingImagePyramid"] = ["MovingRecursiveImagePyramid"] 
    ParamMap['UseDirectionCosines'] = ['true']
    
    if scale is not False:
        ParamMap['Scales'] = [scale]
        ParamMap['AutomaticScalesEstimation'] = ['false']

    # Set the parameter map:
    elastixImageFilter.SetParameterMap(ParamMap) 

    # Register the 2D images:
    elastixImageFilter.Execute()

    # Get the registered image:
    RegIm = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage())
    params = elastixImageFilter.GetTransformParameterMap()[0]["TransformParameters"]
   
    return RegIm, params


# register posteriors using SimpleElastix
def exp_reg_pos_sitk(angle, transform, n_trees=10, n_samples_source=200):
    if transform == 0:
        # rigid transformation
        trans = 'EulerTransform'
        scale = '1'
    else:
        # non-regid transformation
        trans = 'BSplineTransform'
        scale = False
    if angle >= 60 and angle <= 120:
        # use more scales & iterations for large degree transformations
        ite = '2500'
        res = '6'
    else:
        ite = '1500'
        res = '4'
        
    [train_x1, train_x2_rot,test_x1,test_y1,test_x2_rot,test_y2], l2f, uf =\
        get_data(angle, transform, n_samples_source, n_trees)
    _, _, inside, test_grid, _ = to_grid_in_cir(train_x1, train_x2_rot)
    
    # L2F in-task posteriors
    l2f_task1_pos = generate_posteriors(test_grid, 0, l2f, [0,1])
    l2f_task2_pos = generate_posteriors(test_grid, 1, l2f, [0,1])
    task1pred1_reshaped = reshape_posteriors(inside, l2f_task1_pos[0][:,0])
    task2pred2_reshaped = reshape_posteriors(inside, l2f_task2_pos[1][:,0])

    errors = np.zeros(6, dtype=float)
    # UF
    uf_task1 = uf.predict(test_x1, transformer_ids=[0], task_id=0)
    uf_task2 = uf.predict(test_x2_rot, transformer_ids=[1], task_id=1)
    errors[0] = 1 - np.mean(uf_task1 == test_y1)
    errors[1] = 1 - np.mean(uf_task2 == test_y2)

    # L2F
    l2f_task1 = l2f.predict(test_x1, task_id=0)
    l2f_task2 = l2f.predict(test_x2_rot, task_id=1)
    errors[2] = 1 - np.mean(l2f_task1 == test_y1)
    errors[3] = 1 - np.mean(l2f_task2 == test_y2)

    # run elastix
    # morph from task2pred2_reshaped to task1pred1_deformed
    task1pred1_deformed, p1 = run_elastix(
            task1pred1_reshaped, task2pred2_reshaped, res, ite, scale, trans
    )
    # morph from task1pred1_deformed to task2pred2_deformed
    task2pred2_deformed, p2 = run_elastix(
            task2pred2_reshaped, task1pred1_reshaped, res, ite, scale, trans
    )

    if transform == 0:
        # use the transformation learnt from SimpleElastix to transform test points
        test_x2_trans = trans_pts_rigid(p1, test_x2_rot)
        test_x1_trans = trans_pts_rigid(p2, test_x1)
        l2f_task1_pos = generate_posteriors(test_x1, 0, l2f, [0,1])[0]
        l2f_task2_pos = generate_posteriors(test_x2_rot, 1, l2f, [0,1])[1]
        l2f_task1_pos_trans = generate_posteriors(test_x1_trans, 1, l2f, [0,1])[1]
        l2f_task2_pos_trans = generate_posteriors(test_x2_trans, 0, l2f, [0,1])[0]
        w = 1
        sitk_task1 = np.average([l2f_task1_pos, l2f_task1_pos_trans], axis=0, weights=[w,1])
        sitk_task2 = np.average([l2f_task2_pos, l2f_task2_pos_trans], axis=0, weights=[w,1])
    else:
        # assign posteriors for test points using the posteriors of the data points
        # on the test grid that were mapped to the same leaf node
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sitk_task1 = pred_on_task(l2f, 0, test_grid, test_x1, n_trees, task1pred1_deformed, inside)
            sitk_task2 = pred_on_task(l2f, 1, test_grid, test_x2_rot, n_trees, task2pred2_deformed, inside)

    errors[4] = 1 - np.mean(np.argmax(sitk_task1, axis=1) == test_y1)
    errors[5] = 1 - np.mean(np.argmax(sitk_task2, axis=1) == test_y2)

    return errors

## VISUALIZATION FUNCTIONS ##--------------------------------------------------------------------------------------

#Visualize RXOR distribution
def view_rxor(theta=np.pi/4):
    X_xor, y_xor = generate_gaussian_parity(1000)
    X_rxor, y_rxor = generate_gaussian_parity(1000, angle_params=theta)

    colors = sns.color_palette('Dark2', n_colors=2)
    fig, ax = plt.subplots(1,2, figsize=(16,8))

    clr = [colors[i] for i in y_xor]
    ax[0].scatter(X_xor[:, 0], X_xor[:, 1], c=clr, s=50)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Gaussian XOR', fontsize=30)
    ax[0].axis('off')
    
    clr = [colors[i] for i in y_rxor]
    ax[1].scatter(X_rxor[:, 0], X_rxor[:, 1], c=clr, s=50)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Gaussian RXOR', fontsize=30)
    ax[1].axis('off')

    plt.tight_layout()

    
#Visualize SXOR distribution
def view_sxor(shear=0.5):
    X_xor, y_xor = generate_gaussian_parity(1000)
    X_rxor, y_rxor = generate_gaussian_parity(1000)
    X_sxor, y_sxor = shearX(X_rxor, s=shear), y_rxor

    colors = sns.color_palette('Dark2', n_colors=2)
    fig, ax = plt.subplots(1,2, figsize=(16,8))

    clr = [colors[i] for i in y_xor]
    ax[0].scatter(X_xor[:, 0], X_xor[:, 1], c=clr, s=50)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Gaussian XOR', fontsize=30)
    ax[0].axis('off')

    clr = [colors[i] for i in y_sxor]
    ax[1].scatter(X_sxor[:, 0], X_sxor[:, 1], c=clr, s=50)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Gaussian SXOR', fontsize=30)
    ax[1].axis('off')

    plt.tight_layout()
    
    
#Visualize SSXOR distribution
def view_ssxor(ss=(0.5,-0.5)):
    X_xor, y_xor = generate_gaussian_parity(1000)
    X_rxor, y_rxor = generate_gaussian_parity(1000)
    X_ssxor, y_ssxor = double_shearX(X_rxor, y_rxor, ss=ss)

    colors = sns.color_palette('Dark2', n_colors=2)
    fig, ax = plt.subplots(1,2, figsize=(16,8))

    clr = [colors[i] for i in y_xor]
    ax[0].scatter(X_xor[:, 0], X_xor[:, 1], c=clr, s=50)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Gaussian XOR', fontsize=30)
    ax[0].axis('off')

    clr = [colors[i] for i in y_ssxor]
    ax[1].scatter(X_ssxor[:, 0], X_ssxor[:, 1], c=clr, s=50)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Gaussian SSXOR', fontsize=30)
    ax[1].axis('off')

    plt.tight_layout()

    
#Visualize TXOR distribution
def view_txor(t=0.4):
    X_xor, y_xor = generate_gaussian_parity(1000)
    X_rxor, y_rxor = generate_gaussian_parity(1000)
    X_txor, y_txor = div_translateX(X_rxor, y_rxor, t=t)

    colors = sns.color_palette('Dark2', n_colors=2)
    fig, ax = plt.subplots(1,2, figsize=(16,8))

    clr = [colors[i] for i in y_xor]
    ax[0].scatter(X_xor[:, 0], X_xor[:, 1], c=clr, s=50)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Gaussian XOR', fontsize=30)
    ax[0].axis('off')

    clr = [colors[i] for i in y_txor]
    ax[1].scatter(X_txor[:, 0], X_txor[:, 1], c=clr, s=50)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Gaussian TXOR', fontsize=30)
    ax[1].axis('off')

    plt.tight_layout()


# sample random data from uniform distribution
def get_random_pt(xlim_, ylim_):
    x = random.uniform(xlim_[0], xlim_[1])
    y = random.uniform(ylim_[0], ylim_[1])
    return x,y


# make 2 elliptical rings outlined by 3 ellipses
def generate_rand_ellipse(n_sample):
    task_n_sample = n_sample / 2
    data = []
    label = []
    max_ite = 10000
    # arbitrarily define the outer-most ellipse to be x^2/6^2 + y^2/3^2 = 1
    xlim_ = [-6,6]
    ylim_ = [-3,3]
    ite = 0
    # arbitrarily determine the width of the outer elliptical ring to be 1
    # then need to calculate the width of the inner ring in order to
    # keep the areas of the 2 rings to be the same:
    # 5*2\pi - (6*3\pi - 5*2\pi) = 2\pi --> (5-x)(2-x) = 2
    width = np.roots([1,-7,8])[1]  # width of the inner ring

    # make sure there are the same number of data points in either ring
    while (label.count(0) < task_n_sample) or (label.count(1) < task_n_sample):
        if ite < max_ite:
            # use objection sampling: 
            # generate random points and save those lie in a predefined region
            x,y = get_random_pt(xlim_, ylim_)
            if label.count(0) < task_n_sample:
                # outer ring
                if (x**2 / 6**2 + y**2 / 3**2 < 1) & (x**2 / 5**2 + y**2 / 2**2 > 1):
                    data.append([x,y])
                    label.append(0)
            if label.count(1) < task_n_sample:
                # inner ring
                if (x**2 / (5-width)**2 + y**2 / (2-width)**2 > 1) & (x**2 / 5**2 + y**2 / 2**2 < 1):
                    data.append([x,y])
                    label.append(1)
            ite += 1
        else:
            break
    return np.array(data), np.array(label)


#Visualize rotated elliptical rings
def view_rEllip(angle=65):
    X_ellip, y_ellip = generate_rand_ellipse(200)
    X_rEllip, y_rEllip = generate_rand_ellipse(200)
    X_rEllip = rotate(X_rEllip, angle)
    
    colors = sns.color_palette('Dark2', n_colors=2)
    fig, ax = plt.subplots(1,2, figsize=(16,8))

    clr = [colors[i] for i in y_ellip]
    ax[0].scatter(X_ellip[:, 0], X_ellip[:, 1], c=clr, s=50)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Elliptical Rings', fontsize=30)
    ax[0].axis('off')
    
    clr = [colors[i] for i in y_rEllip]
    ax[1].scatter(X_rEllip[:, 0], X_rEllip[:, 1], c=clr, s=50)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Rotated Elliptical Rings', fontsize=30)
    ax[1].axis('off')

    plt.tight_layout()
   

#Visualize sheared elliptical rings
def view_sEllip(angle=65):
    X_ellip, y_ellip = generate_rand_ellipse(200)
    X_rEllip, y_rEllip = generate_rand_ellipse(200)
    X_sEllip, y_sEllip = shearX(X_rEllip, np.tan(angle)), y_rEllip
    
    colors = sns.color_palette('Dark2', n_colors=2)
    fig, ax = plt.subplots(1,2, figsize=(16,8))

    clr = [colors[i] for i in y_ellip]
    ax[0].scatter(X_ellip[:, 0], X_ellip[:, 1], c=clr, s=50)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Elliptical Rings', fontsize=30)
    ax[0].axis('off')

    clr = [colors[i] for i in y_sEllip]
    ax[1].scatter(X_sEllip[:, 0], X_sEllip[:, 1], c=clr, s=50)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Sheared Elliptical Rings', fontsize=30)
    ax[1].axis('off')

    plt.tight_layout()
    
    
#Visualize doubled sheared elliptical rings
def view_ssEllip(ss=(np.tan(65), np.tan(-65))):
    X_ellip, y_ellip = generate_rand_ellipse(200)
    X_rEllip, y_rEllip = generate_rand_ellipse(200)
    X_ssEllip, y_ssEllip = double_shearX(X_rEllip, y_rEllip, ss)
    
    colors = sns.color_palette('Dark2', n_colors=2)
    fig, ax = plt.subplots(1,2, figsize=(16,8))

    clr = [colors[i] for i in y_ellip]
    ax[0].scatter(X_ellip[:, 0], X_ellip[:, 1], c=clr, s=50)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Elliptical Rings', fontsize=30)
    ax[0].axis('off')

    clr = [colors[i] for i in y_ssEllip]
    ax[1].scatter(X_ssEllip[:, 0], X_ssEllip[:, 1], c=clr, s=50)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Sheared Elliptical Rings', fontsize=30)
    ax[1].axis('off')

    plt.tight_layout()
    
def visualize_XOR_transform(Rotation=False, Shear=False, Nonlinear=False, Translate=False):
    cmap_light = ListedColormap(['#FFBBBB', '#BBFFBB', '#BBBBFF'])
    cmap_bold = ListedColormap(['#CC0000', '#00AA00', '#0000CC'])

    #Grid Setup
    l = 3
    h = 0.05 
    xx, yy = np.meshgrid(np.arange(-l, l, h), np.arange(-l, l, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    #distribution generation
    X = [];
    y = [];
    
    if Rotation: 
        S = [30, 45, 60, 90]

        for i in S:
            D,c = generate_gaussian_parity(100, angle_params=i);
    
            X.append(D);
            y.append(c);
    
    elif Shear:
        S = [1, 2, 5, 10]

        for i in S:
            D,c = generate_gaussian_parity(100);
            D = shearX(D, s=i)
    
            X.append(D);
            y.append(c);
            
    elif Nonlinear:
        S = [1, 2, 5, 10]
    
        for i in S:
            D,c = generate_gaussian_parity(100);
            D,c = double_shearX(D,c, ss=(i,-i));
    
            X.append(D);
            y.append(c);
            
    elif Translate:
        S = [0.5, 1, 1.5, 2]
        
        for i in S:
            D,c = generate_gaussian_parity(100);
            D,c = div_translateX(D, c, t=i);
    
            X.append(D);
            y.append(c);

    #Original XOR
    U,v = generate_gaussian_parity(100);
    
    #Prarameters
    n_trees=10
    max_depth=None

    c_afn = [];
    p_afn = [];
    x_afn = [];

    for i in range(len(S)):
        #Model
        default_transformer_class = TreeClassificationTransformer
        default_transformer_kwargs = {"kwargs" : {"max_depth" : max_depth}}

        default_voter_class = TreeClassificationVoter
        default_voter_kwargs = {}

        default_decider_class = SimpleArgmaxAverage
        default_decider_kwargs = {"classes" : np.arange(2)}
        progressive_learner = ProgressiveLearner(
            default_transformer_class = default_transformer_class,
            default_transformer_kwargs = default_transformer_kwargs,
            default_voter_class = default_voter_class,
            default_voter_kwargs = default_voter_kwargs,
            default_decider_class = default_decider_class,
            default_decider_kwargs = default_decider_kwargs)

        #Adaptation
        x = cpd_reg(X[i], U)
        
        if Nonlinear:
            x = nlr_reg(x, U, beta=1)
        if Translate:
            x = nlr_reg(x, U, beta=1)
    
        #Training and Prediction
        progressive_learner.add_task(U, v, num_transformers=n_trees)
        progressive_learner.add_task(X[i], y[i], num_transformers=n_trees)
    
        z = progressive_learner.predict(grid, task_id=0)
        q = progressive_learner.task_id_to_decider[0].predict_proba(grid)[:,0]
    
        #Store values
        c_afn.append(z)
        p_afn.append(q)
        x_afn.append(x)
    
    #Plot Decisions
    l = 2;
    w = 4;
    n = len(S)
    plt.figure(figsize=(w*7, n*7))

    for i in range(n):
        #Decision Boundary
        dnl = c_afn[i];
        dnl = dnl.reshape(xx.shape);
    
        #Posteriors
        pnl = p_afn[i];
        pnl = pnl.reshape(xx.shape);
    
        #Task 2 Distribution
        x = x_afn[i];
        x_orig = X[i]
    
        plt.subplot(n,w, w*i+1);
        plt.scatter(x_orig[:,0], x_orig[:,1], c=y[i], cmap=cmap_bold);
        if Rotation:
            plt.xlim([-l,l]); plt.ylim([-l,l]);
        plt.grid(); plt.title('Task 2');
    
        plt.subplot(n,w, w*i+2);
        plt.scatter(x[:,0], x[:,1], c=y[i], cmap=cmap_bold);
        plt.xlim([-l,l]); plt.ylim([-l,l]);
        plt.grid(); plt.title('Adapted Task 2');  
    
        plt.subplot(n,w, w*i+3);
        plt.pcolormesh(xx, yy, dnl, cmap=cmap_light);
        plt.scatter(U[:,0], U[:,1], c=v, cmap=cmap_bold);
        plt.xlim([-l,l]); plt.ylim([-l,l]);
        plt.title('Combined Decision Boundaries');
    
        plt.subplot(n,w, w*i+4);
        plt.pcolormesh(xx, yy, pnl);
        plt.xlim([-l,l]); plt.ylim([-l,l]);
        plt.title('Combined Posteriors');
    



   
## EXPERIMENT FUNCTIONS ##--------------------------------------------------------------------------------------
def classifier_setup(max_depth=None):
    default_transformer_class = TreeClassificationTransformer
    default_transformer_kwargs = {"kwargs": {"max_depth": max_depth}}

    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = {}

    default_decider_class = SimpleArgmaxAverage
    default_decider_kwargs = {"classes": np.arange(2)}
    progressive_learner = ProgressiveLearner(
        default_transformer_class=default_transformer_class,
        default_transformer_kwargs=default_transformer_kwargs,
        default_voter_class=default_voter_class,
        default_voter_kwargs=default_voter_kwargs,
        default_decider_class=default_decider_class,
        default_decider_kwargs=default_decider_kwargs,
    )
    uf = ProgressiveLearner(
        default_transformer_class=default_transformer_class,
        default_transformer_kwargs=default_transformer_kwargs,
        default_voter_class=default_voter_class,
        default_voter_kwargs=default_voter_kwargs,
        default_decider_class=default_decider_class,
        default_decider_kwargs=default_decider_kwargs,
    )
    naive_uf = ProgressiveLearner(
        default_transformer_class=default_transformer_class,
        default_transformer_kwargs=default_transformer_kwargs,
        default_voter_class=default_voter_class,
        default_voter_kwargs=default_voter_kwargs,
        default_decider_class=default_decider_class,
        default_decider_kwargs=default_decider_kwargs,
    )
    
    return progressive_learner, uf, naive_uf


# generate in-task and cross-task posteriors
def generate_posteriors(X, task_id, forest, transformers):
    vote_per_transformer_id = []
    for transformer_id in transformers:
        vote_per_bag_id = []
        for bag_id in range(
            len(forest.task_id_to_decider[task_id].transformer_id_to_transformers_[transformer_id])
        ):
            transformer = forest.task_id_to_decider[task_id].transformer_id_to_transformers_[transformer_id][
                bag_id
            ]
            X_transformed = transformer.transform(X)
            voter = forest.task_id_to_decider[task_id].transformer_id_to_voters_[transformer_id][bag_id]
            vote = voter.predict_proba(X_transformed)
            vote_per_bag_id.append(vote)
        vote_per_transformer_id.append(np.mean(vote_per_bag_id, axis=0))

    return vote_per_transformer_id


# retrieve training and testing data
def get_data(angle, transform, n_samples_source, n_trees):
    angle = np.pi*angle/180
    train_x1, train_y1 = generate_rand_ellipse(n_samples_source)
    train_x2, train_y2 = generate_rand_ellipse(n_samples_source)
    test_x1, test_y1 = generate_rand_ellipse(n_samples_source)
    test_x2, test_y2 = generate_rand_ellipse(n_samples_source)
    
    if transform == 0:  # rotation (rigid)
        train_x2_rot = rotate(train_x2, angle)
        test_x2_rot = rotate(test_x2, angle)
    elif transform == 1:  # shear (affine)
        train_x2_rot = shearX(train_x2.copy(), np.tan(angle))
        test_x2_rot = shearX(test_x2.copy(), np.tan(angle))
    elif transform == 2:  # double shear (non-linear)
        train_x2_rot, train_y2 = double_shearX(train_x2.copy(), train_y2.copy(), ss=(np.tan(angle), np.tan(-angle)))
        test_x2_rot, test_y2 = double_shearX(test_x2.copy(), test_y2.copy(), ss=(np.tan(angle), np.tan(-angle)))
    data = [train_x1, train_x2_rot, test_x1, test_y1, test_x2_rot, test_y2]
    
    max_depth = ceil(log2(n_samples_source))
    l2f, uf, _ = classifier_setup(max_depth)
    l2f.add_task(train_x1, train_y1, num_transformers=n_trees)
    l2f.add_task(train_x2_rot, train_y2, num_transformers=n_trees)
    uf.add_task(train_x1, train_y1, num_transformers=2*n_trees)
    uf.add_task(train_x2_rot, train_y2, num_transformers=2*n_trees)
    return data, l2f, uf


#Rotation Experiment------------------------------------------------------------------------------------------------
def experiment_rxor(
    n_task1,
    n_task2,
    n_test=1000, 
    task1_angle=0,
    task2_angle=np.pi/2, 
    n_trees=10,
    max_depth=None,
    random_state=None,
    register_cpd=False,
    register_nlr=False,
    register_otp=False,
    register_icp=False,
    bte=True,
):

    if n_task1 == 0 and n_task2 == 0:
        raise ValueError("Wake up and provide samples to train!!!")
    if random_state != None:
        np.random.seed(random_state)
    
    #error array
    errors = np.zeros(6, dtype=float)

    #classifier setup
    progressive_learner, uf, naive_uf = classifier_setup(max_depth=max_depth)

    #task 1 data
    X_task1, y_task1 = generate_gaussian_parity(n_task1, angle_params=task1_angle)
    test_task1, test_label_task1 = generate_gaussian_parity(n_test, angle_params=task1_angle)

    #task 2 data
    X_task2, y_task2 = generate_gaussian_parity(n_task2, angle_params=task2_angle)
    test_task2, test_label_task2 = generate_gaussian_parity(n_test, angle_params=task2_angle)
    
    #registration
    if register_cpd:
        if bte:
            X_task2 = cpd_reg(X_task2.copy(), X_task1.copy())
        else:
            X_task1 = cpd_reg(X_task1.copy(), X_task2.copy())
            
    if register_nlr:
        if bte:
            X_task2 = nlr_reg(X_task2.copy(), X_task1.copy())
        else:
            X_task1 = nlr_reg(X_task1.copy(), X_task2.copy())
        
    if register_icp:
        if bte:
            T, X_3, i = icp(X_task2.copy(), X_task1.copy(), y_task2.copy(), y_task1.copy())
            X_task2 = X_3.T[:, 0:2]
        else:
            T, X_3, i = icp(X_task1.copy(), X_task2.copy(), y_task1.copy(), y_task2.copy())
            X_task1 = X_3.T[:, 0:2]

    #train and predict
    progressive_learner.add_task(X_task1, y_task1, num_transformers=n_trees)
    progressive_learner.add_task(X_task2, y_task2, num_transformers=n_trees)

    uf.add_task(X_task1, y_task1, num_transformers=2 * n_trees)
    uf.add_task(X_task2, y_task2, num_transformers=2 * n_trees)

    if bte:
        uf_task1 = uf.predict(test_task1, transformer_ids=[0], task_id=0)
        l2f_task1 = progressive_learner.predict(test_task1, task_id=0)
        errors[0] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[1] = 1 - np.mean(l2f_task1 == test_label_task1)
    else:
        uf_task2 = uf.predict(test_task2, transformer_ids=[1], task_id=1)
        l2f_task2 = progressive_learner.predict(test_task2, task_id=1)
        errors[0] = 1 - np.mean(uf_task2 == test_label_task2)
        errors[1] = 1 - np.mean(l2f_task2 == test_label_task2)

    return errors

def bte_v_angle(angle_sweep, task1_sample, task2_sample, mc_rep, register_cpd=False, register_nlr=False, register_otp=False, register_icp=False):
    mean_te = np.zeros(len(angle_sweep), dtype=float)
    for ii, angle in enumerate(angle_sweep):
        error = np.array(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(experiment_rxor)(
                    task1_sample,
                    task2_sample,
                    task2_angle=angle*np.pi/180,
                    max_depth=ceil(log2(task1_sample)),
                    register_cpd=register_cpd,
                    register_nlr=register_nlr,
                    register_otp=register_otp,
                    register_icp=register_icp,
                    bte = True
                )
                for _ in range(mc_rep)
            )
        )

        mean_te[ii] = np.mean(error[:, 0]) / np.mean(error[:, 1])

    return mean_te

def fte_v_angle(angle_sweep, task1_sample, task2_sample, mc_rep, register_cpd=False, register_nlr=False, register_otp=False, register_icp=False):
    mean_te = np.zeros(len(angle_sweep), dtype=float)
    for ii, angle in enumerate(angle_sweep):
        error = np.array(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(experiment_rxor)(
                    task1_sample,
                    task2_sample,
                    task2_angle=angle*np.pi/180,
                    max_depth=ceil(log2(task1_sample)),
                    register_cpd=register_cpd,
                    register_nlr=register_nlr,
                    register_otp=register_otp,
                    register_icp=register_icp,
                    bte = False
                )
                for _ in range(mc_rep)
            )
        )

        mean_te[ii] = np.mean(error[:, 0]) / np.mean(error[:, 1])

    return mean_te

def plot_te_v_angle(angle_sweep, btes, ftes):
    colors = sns.color_palette('Dark2', n_colors=5)

    sns.set_context("talk")
    fig = plt.figure(constrained_layout=True, figsize=(25, 15))
    gs = fig.add_gridspec(6, 12)
    ax = fig.add_subplot(gs[:6, :6])
    task = ["No adaptation", "CPD (Affine)", "ICP", "CPD (Nonlinear)"]
    ax.plot(angle_sweep, btes[0], c=colors[0], linewidth=3, label=task[0])
    ax.plot(angle_sweep, btes[1], c=colors[1], linewidth=3, label=task[1])
    ax.plot(angle_sweep, btes[2], c=colors[2], linewidth=3, label=task[2])
    ax.plot(angle_sweep, btes[3], c=colors[3], linewidth=3, label=task[3])
    ax.set_xlabel("Angle of Rotation (Degrees)", fontsize=30)
    ax.set_ylabel("Backward Transfer Efficiency (XOR)", fontsize=30)
    ax.set_xticks(range(0,91,10))
    ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3])
    ax.tick_params(labelsize=24)
    ax.hlines(1, 0, 90, colors="grey", linestyles="dashed", linewidth=1.5)
    ax.legend(loc="lower left", fontsize=20, frameon=False)
    ax.set_title("BTE vs Angle", fontsize=30)

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    
    ax = fig.add_subplot(gs[:6, 6:])
    task = ["No adaptation", "CPD (Affine)", "ICP", "CPD (Nonlinear)"]
    ax.plot(angle_sweep, ftes[0], c=colors[0], linewidth=3, label=task[0])
    ax.plot(angle_sweep, ftes[1], c=colors[1], linewidth=3, label=task[1])
    ax.plot(angle_sweep, ftes[2], c=colors[2], linewidth=3, label=task[2])
    ax.plot(angle_sweep, ftes[3], c=colors[3], linewidth=3, label=task[3])
    ax.set_xlabel("Angle of Rotation (Degrees)", fontsize=30)
    ax.set_ylabel("Forward Transfer Efficiency (XOR)", fontsize=30)
    ax.set_xticks(range(0,91,10))
    ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3])
    ax.tick_params(labelsize=24)
    ax.hlines(1, 0, 90, colors="grey", linestyles="dashed", linewidth=1.5)
    ax.legend(loc="lower left", fontsize=20, frameon=False)
    ax.set_title("FTE vs Angle", fontsize=30)

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    
    
    
#Shear Experiment------------------------------------------------------------------------------------------------
def experiment_sxor(
    n_task1,
    n_task2,
    n_test=1000, 
    task1_angle=0,
    task2_shear=1, 
    n_trees=10,
    max_depth=None,
    random_state=None,
    register_cpd=False,
    register_nlr=False,
    register_otp=False,
    register_icp=False,
    bte=True
):

    if n_task1 == 0 and n_task2 == 0:
        raise ValueError("Wake up and provide samples to train!!!")
    if random_state != None:
        np.random.seed(random_state)

    #error array
    errors = np.zeros(6, dtype=float)

    #classifier setup
    progressive_learner, uf, naive_uf = classifier_setup(max_depth=max_depth)

    #task 1 data
    X_task1, y_task1 = generate_gaussian_parity(n_task1, angle_params=task1_angle)
    test_task1, test_label_task1 = generate_gaussian_parity(n_test, angle_params=task1_angle)

    #task 2 data
    X_task2, y_task2 = generate_gaussian_parity(n_task2, angle_params=task1_angle)
    test_task2, test_label_task2 = generate_gaussian_parity(n_test, angle_params=task1_angle)
    
    #transform task 2
    X_task2 = shearX(X_task2, s=task2_shear)
    test_task2 = shearX(test_task2, s=task2_shear)
    
    #registration
    if register_cpd:
        if bte:
            X_task2 = cpd_reg(X_task2.copy(), X_task1.copy())
        else:
            X_task1 = cpd_reg(X_task1.copy(), X_task2.copy())
            
    if register_nlr:
        if bte:
            X_task2 = nlr_reg(X_task2.copy(), X_task1.copy())
        else:
            X_task1 = nlr_reg(X_task1.copy(), X_task2.copy())
        
    if register_icp:
        if bte:
            T, X_3, i = icp(X_task2.copy(), X_task1.copy(), y_task2.copy(), y_task1.copy())
            X_task2 = X_3.T[:, 0:2]
        else:
            T, X_3, i = icp(X_task1.copy(), X_task2.copy(), y_task1.copy(), y_task2.copy())
            X_task1 = X_3.T[:, 0:2]

    #train and predict
    progressive_learner.add_task(X_task1, y_task1, num_transformers=n_trees)
    progressive_learner.add_task(X_task2, y_task2, num_transformers=n_trees)

    uf.add_task(X_task1, y_task1, num_transformers=2 * n_trees)
    uf.add_task(X_task2, y_task2, num_transformers=2 * n_trees)

    if bte:
        uf_task1 = uf.predict(test_task1, transformer_ids=[0], task_id=0)
        l2f_task1 = progressive_learner.predict(test_task1, task_id=0)
        errors[0] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[1] = 1 - np.mean(l2f_task1 == test_label_task1)
    else:
        uf_task2 = uf.predict(test_task2, transformer_ids=[1], task_id=1)
        l2f_task2 = progressive_learner.predict(test_task2, task_id=1)
        errors[0] = 1 - np.mean(uf_task2 == test_label_task2)
        errors[1] = 1 - np.mean(l2f_task2 == test_label_task2)

    return errors

def bte_v_shear(shear_sweep, task1_sample, task2_sample, mc_rep, register_cpd=False, register_nlr=False, register_otp=False, register_icp=False):
    mean_te = np.zeros(len(shear_sweep), dtype=float)
    for ii, s in enumerate(shear_sweep):
        error = np.array(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(experiment_sxor)(
                    task1_sample,
                    task2_sample,
                    task2_shear=s,
                    max_depth=ceil(log2(task1_sample)),
                    register_cpd=register_cpd,
                    register_nlr=register_nlr,
                    register_otp=register_otp,
                    register_icp=register_icp,
                    bte=True
                )
                for _ in range(mc_rep)
            )
        )

        mean_te[ii] = np.mean(error[:, 0]) / np.mean(error[:, 1])

    return mean_te

def fte_v_shear(shear_sweep, task1_sample, task2_sample, mc_rep, register_cpd=False, register_otp=False, register_icp=False, register_nlr=False):
    mean_te = np.zeros(len(shear_sweep), dtype=float)
    for ii, s in enumerate(shear_sweep):
        error = np.array(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(experiment_sxor)(
                    task1_sample,
                    task2_sample,
                    task2_shear=s,
                    max_depth=ceil(log2(task1_sample)),
                    register_cpd=register_cpd,
                    register_otp=register_otp,
                    register_icp=register_icp,
                    register_nlr=register_nlr,
                    bte=False,
                )
                for _ in range(mc_rep)
            )
        )

        mean_te[ii] = np.mean(error[:, 0]) / np.mean(error[:, 1])

    return mean_te

def plot_te_v_shear(shear_sweep, btes, ftes):
    colors = sns.color_palette('Dark2', n_colors=5)

    sns.set_context("talk")
    fig = plt.figure(constrained_layout=True, figsize=(25, 15))
    gs = fig.add_gridspec(6, 12)
    ax = fig.add_subplot(gs[:6, :6])
    task = ["No adaptation", "CPD (Affine)", "ICP", "CPD (Nonlinear)"]
    ax.plot(shear_sweep, btes[0], c=colors[0], linewidth=3, label=task[0])
    ax.plot(shear_sweep, btes[1], c=colors[1], linewidth=3, label=task[1])
    ax.plot(shear_sweep, btes[2], c=colors[2], linewidth=3, label=task[2])
    ax.plot(shear_sweep, btes[3], c=colors[3], linewidth=3, label=task[3])
    ax.set_xlabel("Shear Value (S)", fontsize=30)
    ax.set_ylabel("Backward Transfer Efficiency (SXOR)", fontsize=30)
    ax.set_xscale('log')
    ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3])
    ax.tick_params(labelsize=24)
    ax.hlines(1, 0, 500, colors="grey", linestyles="dashed", linewidth=1.5)
    ax.legend(loc="lower left", fontsize=20, frameon=False)
    ax.set_title("BTE vs Shear", fontsize=30)

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    
    ax = fig.add_subplot(gs[:6, 6:])
    task = ["No adaptation", "CPD (Affine)", "ICP", "CPD (Nonlinear)"]
    ax.plot(shear_sweep, ftes[0], c=colors[0], linewidth=3, label=task[0])
    ax.plot(shear_sweep, ftes[1], c=colors[1], linewidth=3, label=task[1])
    ax.plot(shear_sweep, ftes[2], c=colors[2], linewidth=3, label=task[2])
    ax.plot(shear_sweep, ftes[3], c=colors[3], linewidth=3, label=task[3])
    ax.set_xlabel("Shear Value (S)", fontsize=30)
    ax.set_ylabel("Forward Transfer Efficiency (SXOR)", fontsize=30)
    ax.set_xscale('log')
    ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3])
    ax.tick_params(labelsize=24)
    ax.hlines(1, 0, 500, colors="grey", linestyles="dashed", linewidth=1.5)
    ax.legend(loc="lower left", fontsize=20, frameon=False)
    ax.set_title("FTE vs Shear", fontsize=30)

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    
#Double Shear Experiment------------------------------------------------------------------------------------------------    
def experiment_ssxor(
    n_task1,
    n_task2,
    n_test=1000, 
    task1_angle=0,
    task2_shear=1, 
    n_trees=10,
    max_depth=None,
    random_state=None,
    register_cpd=False,
    register_nlr=False,
    register_otp=False,
    register_icp=False,
    bte=True
):

    if n_task1 == 0 and n_task2 == 0:
        raise ValueError("Wake up and provide samples to train!!!")
    if random_state != None:
        np.random.seed(random_state)

    #error array
    errors = np.zeros(6, dtype=float)

    #classifier setup
    progressive_learner, uf, naive_uf = classifier_setup(max_depth=max_depth)

    #task 1 data
    X_task1, y_task1 = generate_gaussian_parity(n_task1, angle_params=task1_angle)
    test_task1, test_label_task1 = generate_gaussian_parity(n_test, angle_params=task1_angle)

    #task 2 data
    X_task2, y_task2 = generate_gaussian_parity(n_task2, angle_params=task1_angle)
    test_task2, test_label_task2 = generate_gaussian_parity(n_test, angle_params=task1_angle)
    
    #transform task 2
    X_task2, y_task2 = double_shearX(X_task2, y_task2, ss=(task2_shear, -task2_shear));
    test_task2, test_label_task2 = double_shearX(test_task2, test_label_task2, ss=(task2_shear, -task2_shear));
 
    #registration
    if register_cpd:
        if bte:
            X_task2 = cpd_reg(X_task2.copy(), X_task1.copy())
        else:
            X_task1 = cpd_reg(X_task1.copy(), X_task2.copy())
            
    if register_nlr:
        if bte:
            X_task2 = nlr_reg(X_task2.copy(), X_task1.copy())
        else:
            X_task1 = nlr_reg(X_task1.copy(), X_task2.copy())
        
    if register_icp:
        if bte:
            T, X_3, i = icp(X_task2.copy(), X_task1.copy(), y_task2.copy(), y_task1.copy())
            X_task2 = X_3.T[:, 0:2]
        else:
            T, X_3, i = icp(X_task1.copy(), X_task2.copy(), y_task1.copy(), y_task2.copy())
            X_task1 = X_3.T[:, 0:2]

    #train and predict
    progressive_learner.add_task(X_task1, y_task1, num_transformers=n_trees)
    progressive_learner.add_task(X_task2, y_task2, num_transformers=n_trees)

    uf.add_task(X_task1, y_task1, num_transformers=2 * n_trees)
    uf.add_task(X_task2, y_task2, num_transformers=2 * n_trees)

    if bte:
        uf_task1 = uf.predict(test_task1, transformer_ids=[0], task_id=0)
        l2f_task1 = progressive_learner.predict(test_task1, task_id=0)
        errors[0] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[1] = 1 - np.mean(l2f_task1 == test_label_task1)
    else:
        uf_task2 = uf.predict(test_task2, transformer_ids=[1], task_id=1)
        l2f_task2 = progressive_learner.predict(test_task2, task_id=1)
        errors[0] = 1 - np.mean(uf_task2 == test_label_task2)
        errors[1] = 1 - np.mean(l2f_task2 == test_label_task2)

    return errors

def bte_v_double_shear(shear_sweep, task1_sample, task2_sample, mc_rep, register_cpd=False, register_nlr=False, register_otp=False, register_icp=False):
    mean_te = np.zeros(len(shear_sweep), dtype=float)
    for ii, s in enumerate(shear_sweep):
        error = np.array(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(experiment_ssxor)(
                    task1_sample,
                    task2_sample,
                    task2_shear=s,
                    max_depth=ceil(log2(task1_sample)),
                    register_cpd=register_cpd,
                    register_nlr=register_nlr,
                    register_otp=register_otp,
                    register_icp=register_icp,
                    bte=True
                )
                for _ in range(mc_rep)
            )
        )

        mean_te[ii] = np.mean(error[:, 0]) / np.mean(error[:, 1])

    return mean_te

def fte_v_double_shear(shear_sweep, task1_sample, task2_sample, mc_rep, register_cpd=False, register_nlr=False, register_otp=False, register_icp=False):
    mean_te = np.zeros(len(shear_sweep), dtype=float)
    for ii, s in enumerate(shear_sweep):
        error = np.array(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(experiment_ssxor)(
                    task1_sample,
                    task2_sample,
                    task2_shear=s,
                    max_depth=ceil(log2(task1_sample)),
                    register_cpd=register_cpd,
                    register_nlr=register_nlr,
                    register_otp=register_otp,
                    register_icp=register_icp,
                    bte=False
                )
                for _ in range(mc_rep)
            )
        )

        mean_te[ii] = np.mean(error[:, 0]) / np.mean(error[:, 1])

    return mean_te

def plot_te_v_double_shear(shear_sweep, btes, ftes):
    colors = sns.color_palette('Dark2', n_colors=5)

    sns.set_context("talk")
    fig = plt.figure(constrained_layout=True, figsize=(25, 15))
    gs = fig.add_gridspec(6, 12)
    ax = fig.add_subplot(gs[:6, :6])
    task = ["No adaptation", "CPD (Affine)", "ICP", "CPD (Nonlinear)"]
    ax.plot(shear_sweep, btes[0], c=colors[0], linewidth=3, label=task[0])
    ax.plot(shear_sweep, btes[1], c=colors[1], linewidth=3, label=task[1])
    ax.plot(shear_sweep, btes[2], c=colors[2], linewidth=3, label=task[2])
    ax.plot(shear_sweep, btes[3], c=colors[3], linewidth=3, label=task[3])
    ax.set_xlabel("Shear Value (S)", fontsize=30)
    ax.set_ylabel("Backward Transfer Efficiency (SSXOR)", fontsize=30)
    ax.set_xscale('log')
    ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3])
    ax.tick_params(labelsize=24)
    ax.hlines(1, 0, 500, colors="grey", linestyles="dashed", linewidth=1.5)
    ax.legend(loc="lower left", fontsize=20, frameon=False)
    ax.set_title("BTE vs Double Shear", fontsize=30)

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    
    ax = fig.add_subplot(gs[:6, 6:])
    task = ["No adaptation", "CPD (Affine)", "ICP", "CPD (Nonlinear)"]
    ax.plot(shear_sweep, ftes[0], c=colors[0], linewidth=3, label=task[0])
    ax.plot(shear_sweep, ftes[1], c=colors[1], linewidth=3, label=task[1])
    ax.plot(shear_sweep, ftes[2], c=colors[2], linewidth=3, label=task[2])
    ax.plot(shear_sweep, ftes[3], c=colors[3], linewidth=3, label=task[3])
    ax.set_xlabel("Shear Value (S)", fontsize=30)
    ax.set_ylabel("Forward Transfer Efficiency (SSXOR)", fontsize=30)
    ax.set_xscale('log')
    ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3])
    ax.tick_params(labelsize=24)
    ax.hlines(1, 0, 500, colors="grey", linestyles="dashed", linewidth=1.5)
    ax.legend(loc="lower left", fontsize=20, frameon=False)
    ax.set_title("FTE vs Double Shear", fontsize=30)

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    
#Divergent Translation Experiment------------------------------------------------------------------------------------------------
def experiment_txor(
    n_task1,
    n_task2,
    n_test=1000, 
    task1_angle=0,
    task2_trans=1, 
    n_trees=10,
    max_depth=None,
    random_state=None,
    register_cpd=False,
    register_nlr=False,
    register_otp=False,
    register_icp=False,
    bte=True
):

    if n_task1 == 0 and n_task2 == 0:
        raise ValueError("Wake up and provide samples to train!!!")
    if random_state != None:
        np.random.seed(random_state)

    #error array
    errors = np.zeros(6, dtype=float)

    #classifier setup
    progressive_learner, uf, naive_uf = classifier_setup(max_depth=max_depth)

    #task 1 data
    X_task1, y_task1 = generate_gaussian_parity(n_task1, angle_params=task1_angle)
    test_task1, test_label_task1 = generate_gaussian_parity(n_test, angle_params=task1_angle)

    #task 2 data
    X_task2, y_task2 = generate_gaussian_parity(n_task2, angle_params=task1_angle)
    test_task2, test_label_task2 = generate_gaussian_parity(n_test, angle_params=task1_angle)
    
    #transform task 2
    X_task2, y_task2 = div_translateX(X_task2, y_task2, t=task2_trans);
    test_task2, test_label_task2 = div_translateX(test_task2, test_label_task2, t=task2_trans);
 
    #registration
    if register_cpd:
        if bte:
            X_task2 = cpd_reg(X_task2.copy(), X_task1.copy())
        else:
            X_task1 = cpd_reg(X_task1.copy(), X_task2.copy())
            
    if register_nlr:
        if bte:
            X_task2 = nlr_reg(X_task2.copy(), X_task1.copy())
        else:
            X_task1 = nlr_reg(X_task1.copy(), X_task2.copy())
        
    if register_icp:
        if bte:
            T, X_3, i = icp(X_task2.copy(), X_task1.copy(), y_task2.copy(), y_task1.copy())
            X_task2 = X_3.T[:, 0:2]
        else:
            T, X_3, i = icp(X_task1.copy(), X_task2.copy(), y_task1.copy(), y_task2.copy())
            X_task1 = X_3.T[:, 0:2]

    #train and predict
    progressive_learner.add_task(X_task1, y_task1, num_transformers=n_trees)
    progressive_learner.add_task(X_task2, y_task2, num_transformers=n_trees)

    uf.add_task(X_task1, y_task1, num_transformers=2 * n_trees)
    uf.add_task(X_task2, y_task2, num_transformers=2 * n_trees)

    if bte:
        uf_task1 = uf.predict(test_task1, transformer_ids=[0], task_id=0)
        l2f_task1 = progressive_learner.predict(test_task1, task_id=0)
        errors[0] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[1] = 1 - np.mean(l2f_task1 == test_label_task1)
    else:
        uf_task2 = uf.predict(test_task2, transformer_ids=[1], task_id=1)
        l2f_task2 = progressive_learner.predict(test_task2, task_id=1)
        errors[0] = 1 - np.mean(uf_task2 == test_label_task2)
        errors[1] = 1 - np.mean(l2f_task2 == test_label_task2)

    return errors

def bte_v_translate(trans_sweep, task1_sample, task2_sample, mc_rep, register_cpd=False, register_nlr=False, register_otp=False, register_icp=False):
    mean_te = np.zeros(len(trans_sweep), dtype=float)
    for ii, t in enumerate(trans_sweep):
        error = np.array(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(experiment_txor)(
                    task1_sample,
                    task2_sample,
                    task2_trans=t,
                    max_depth=ceil(log2(task1_sample)),
                    register_cpd=register_cpd,
                    register_nlr=register_nlr,
                    register_otp=register_otp,
                    register_icp=register_icp,
                    bte=True
                )
                for _ in range(mc_rep)
            )
        )

        mean_te[ii] = np.mean(error[:, 0]) / np.mean(error[:, 1])

    return mean_te

def fte_v_translate(trans_sweep, task1_sample, task2_sample, mc_rep, register_cpd=False, register_nlr=False, register_otp=False, register_icp=False):
    mean_te = np.zeros(len(trans_sweep), dtype=float)
    for ii, t in enumerate(trans_sweep):
        error = np.array(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(experiment_txor)(
                    task1_sample,
                    task2_sample,
                    task2_trans=t,
                    max_depth=ceil(log2(task1_sample)),
                    register_cpd=register_cpd,
                    register_nlr=register_nlr,
                    register_otp=register_otp,
                    register_icp=register_icp,
                    bte=False
                )
                for _ in range(mc_rep)
            )
        )

        mean_te[ii] = np.mean(error[:, 0]) / np.mean(error[:, 1])

    return mean_te

def plot_bte_v_translate(trans_sweep, btes, ftes):
    colors = sns.color_palette('Dark2', n_colors=5)

    sns.set_context("talk")
    fig = plt.figure(constrained_layout=True, figsize=(25, 15))
    gs = fig.add_gridspec(6, 12)
    ax = fig.add_subplot(gs[:6, :6])
    task = ["No adaptation", "CPD (Affine)", "ICP", "CPD (Nonlinear)"]
    ax.plot(trans_sweep, btes[0], c=colors[0], linewidth=3, label=task[0])
    ax.plot(trans_sweep, btes[1], c=colors[1], linewidth=3, label=task[1])
    ax.plot(trans_sweep, btes[2], c=colors[2], linewidth=3, label=task[2])
    ax.plot(trans_sweep, btes[3], c=colors[3], linewidth=3, label=task[3])
    ax.set_xlabel("Translate Value (T)", fontsize=30)
    ax.set_ylabel("Backward Transfer Efficiency (TXOR)", fontsize=30)
    ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3])
    ax.tick_params(labelsize=24)
    ax.hlines(1, 0, 2, colors="grey", linestyles="dashed", linewidth=1.5)
    ax.legend(loc="lower left", fontsize=20, frameon=False)
    ax.set_title("BTE vs Divergent Translation", fontsize=30)

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    
    ax = fig.add_subplot(gs[:6, 6:])
    task = ["No adaptation", "CPD (Affine)", "ICP", "CPD (Nonlinear)"]
    ax.plot(trans_sweep, ftes[0], c=colors[0], linewidth=3, label=task[0])
    ax.plot(trans_sweep, ftes[1], c=colors[1], linewidth=3, label=task[1])
    ax.plot(trans_sweep, ftes[2], c=colors[2], linewidth=3, label=task[2])
    ax.plot(trans_sweep, ftes[3], c=colors[3], linewidth=3, label=task[3])
    ax.set_xlabel("Translate Value (T)", fontsize=30)
    ax.set_ylabel("Forward Transfer Efficiency (TXOR)", fontsize=30)
    ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3])
    ax.tick_params(labelsize=24)
    ax.hlines(1, 0, 2, colors="grey", linestyles="dashed", linewidth=1.5)
    ax.legend(loc="lower left", fontsize=20, frameon=False)
    ax.set_title("FTE vs Divergent Translation", fontsize=30)

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)


#Registering-Posteriors Experiment------------------------------------------------------------------------------------------------
def te_v_angle_reg_pos(n_trees=10, n_samples_source=200, rep=1000):
    # store the raw generalization errors
    errors_all = {}
    angles_all = [np.linspace(0,180,13), 
                  np.delete(np.linspace(0,180,13), 6), np.delete(np.linspace(0,180,13), 6)]

    def _run(transform, alg):
        angles = angles_all[transform]
        errors = np.empty((len(angles), 6))
        for i,angle in enumerate(angles):
            if alg == 'OT':
                errors[i,:] = exp_reg_pos_OT(angle, transform, n_trees, n_samples_source)
            elif alg == 'SE':
                errors[i,:] = exp_reg_pos_sitk(angle, transform, n_trees, n_samples_source)
        return errors

    # run the experiment for all 3 types of transformations
    for transform in range(3):
        # using OT
        alg = 'OT'
        errors_all[alg + '-' + str(t)] = Parallel(n_jobs=35, verbose=1)(
            delayed(_run)(transform, alg) for _ in range(rep)
        )
        # using SE
        alg = 'SE'
        errors_all[alg + '-' + str(t)] = Parallel(n_jobs=35, verbose=1)(
            delayed(_run)(transform, alg) for _ in range(rep)
        )
        
    # compute FTE and BTE using generalization errors for each algorithm
    def _error_to_te(e_avg, n_angles=13):
        mean_te_all = []
        for n_alg in range(2):
            e_single = np.hstack((e_avg[:,:2], e_avg[:,n_alg*2+2:n_alg*2+4]))
            mean_te = np.zeros((2, n_angles))
            for i in range(mean_te.shape[0]):
                for j in range(mean_te.shape[1]):
                    mean_te[i,j] = e_single[j,i] / e_single[j,i+2]
            mean_te_all.append(mean_te)
        return mean_te_all

    TEs_all = {}
    n_angles = [13, 12, 12]
    for transform in range(3):
        TEs_all[transform] = []
        error_ = [np.mean(errors_all['OT-'+str(transform)], axis=0),
                 np.mean(errors_all['SE-'+str(transform)], axis=0)]
        for alg in range(len(error_)):
            TEs_all[transform].append(error_to_te(error_[alg], n_angles[transform]))
            
    return TEs_all


def plot_te_v_angle_res_pos(TEs_all):
    colors = ['g','r','b']
    fontsize=30; labelsize=28
    fig, axs = plt.subplots(2,3, figsize=(30,14))
    angles = [np.linspace(0,180,13), 
              np.delete(np.linspace(0,180,13), 6), np.delete(np.linspace(0,180,13), 6)]
    ylabels = ['log BTE', 'log FTE']
    titles = ['Rigid', 'Affine', 'Nonlinear']
    labels = ['O_DIF', 'O_DIF o OptimalTransport', 'O_DIF o SimpleElastix']
    line_styles = ['dashed', '-', '-']
    for i in range(3):
        data = []
        data.append(TEs_all[i][0][0].copy())
        for ii in range(len(TEs_all[i])):
            data.append(TEs_all[i][ii][1].copy())
        for j in range(2):
            ax = axs[j,i]
            for data_idx in range(len(data)):
                ax.plot(
                    angles[i], np.log(data[data_idx][j]),label=labels[data_idx],
                    lw=3, c=colors[data_idx], ls=line_styles[data_idx]
                )
                ax.set_xticks(angles[i])
                ax.set_xticklabels(angles[i].astype(int), rotation=45)
                if i == 0:
                    ax.set_ylabel(ylabels[j], fontsize=fontsize)
                ax.tick_params(labelsize=labelsize)
                ax.hlines(0, angles[i][0], angles[i][-1], colors='gray', linestyles='dashed',linewidth=1.5)
                right_side = ax.spines["right"]
                right_side.set_visible(False)
                top_side = ax.spines["top"]
                top_side.set_visible(False)
                if j == 0:
                    ax.set_xticklabels("")
                    ax.set_title(titles[i], fontsize=fontsize)
                else:
                    if i == 1:
                        ax.set_xlabel('Angle of Transformation (Degrees)', fontsize=fontsize)
                    if i == 2:
                        ax.legend(loc='upper center', fontsize=20, frameon=False)


def visualize_ellip_transform(angle=45, n_samples_source=200, n_trees=10):
    fig,axs = plt.subplots(3,4, figsize=(16,12))
    colors = sns.color_palette('Dark2', n_colors=5)
    cmap_light = ListedColormap(['#FFBBBB', '#BBFFBB', '#BBBBFF'])
    cmap_bold = ListedColormap(['#CC0000', '#00AA00', '#0000CC'])
    titles = ['Task 1','Adapted Task1', 'Task 2', 'Adapted Task 2']
    ylabels = ['Rigid', 'Affine', 'Nonlinear']
    xylim = [-7, 7]
    for transform in range(3):
        [train_x1, train_x2_rot, test_x1, _, test_x2_rot, _], l2f, _ =\
            get_data(angle, transform, n_samples_source, n_trees)   
        reg = [1.0,1.0,1.0]
        if transform == 0:
            # SP
            SP = SeedlessProcrustes(optimal_transport_lambda=reg[transform], optimal_transport_eps=10e-15,
                                   optimal_transport_num_reps=10)
            SP.fit(train_x1, train_x2_rot)
            test_x1_trans = SP.transform(test_x1)

            SP = SeedlessProcrustes(optimal_transport_lambda=reg[transform], optimal_transport_eps=10e-15,
                                   optimal_transport_num_reps=10)
            SP.fit(train_x2_rot, train_x1)
            test_x2_trans = SP.transform(test_x2_rot)
        else:
            # OT
            OT = ot.da.SinkhornTransport(reg_e=0.15, tol=10e-15)
            OT.fit(Xs=train_x1, Xt=train_x2_rot)
            test_x1_trans = OT.transform(Xs=test_x1)
            test_x2_trans = OT.inverse_transform(Xt=test_x2_rot)
        data = [test_x1, test_x1_trans, test_x2_rot, test_x2_trans]
        # generate posteriors
        l2f_task1_pos = generate_posteriors(test_x1, 0, l2f, [0,1])[0]
        l2f_task2_pos = generate_posteriors(test_x2_rot, 1, l2f, [0,1])[1]
        l2f_task1_pos_trans = generate_posteriors(test_x1_trans, 1, l2f, [0,1])[1]
        l2f_task2_pos_trans = generate_posteriors(test_x2_trans, 0, l2f, [0,1])[0]
        labels = [l2f_task1_pos, l2f_task1_pos_trans, l2f_task2_pos, l2f_task2_pos_trans]
        for i in range(len(data)):
            ax = axs[transform, i]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.scatter(data[i][:,0], data[i][:,1], c=labels[i][:,0], s=10)
            ax.set_xlim(xylim); ax.set_ylim(xylim)
            if i == 0:
                ax.set_ylabel(ylabels[transform], fontsize=20)
            if transform == 0:
                ax.set_title(titles[i], fontsize=20)
