# Import the packages for experiment
import warnings

warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import random
from skimage.transform import rotate
from scipy import ndimage
from skimage.util import img_as_ubyte
import numpy as np
import seaborn as sns

# Import the progressive learning packages
from proglearn.forest import LifelongClassificationForest

# Randomized selection of training and testing subsets
def cross_val_data(data_x, data_y, total_cls=10):
    # Creates copies of both data_x and data_y so that they can be modified without affecting the original sets
    x = data_x.copy()
    y = data_y.copy()
    # Creates a sorted array of arrays that each contain the indices at which each unique element of data_y can be found
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]

    for i in range(total_cls):
        # Chooses the i'th array within the larger idx array
        indx = idx[i]
        # The elements of indx are randomly shuffled
        random.shuffle(indx)

        if i == 0:
            # 250 training data points per task
            train_x1 = x[indx[0:250], :]
            train_x2 = x[indx[250:500], :]
            train_y1 = y[indx[0:250]]
            train_y2 = y[indx[250:500]]

            # 100 testing data points per task
            test_x = x[indx[500:600], :]
            test_y = y[indx[500:600]]
        else:
            # 250 training data points per task
            train_x1 = np.concatenate((train_x1, x[indx[0:250], :]), axis=0)
            train_x2 = np.concatenate((train_x2, x[indx[250:500], :]), axis=0)
            train_y1 = np.concatenate((train_y1, y[indx[0:250]]), axis=0)
            train_y2 = np.concatenate((train_y2, y[indx[250:500]]), axis=0)

            # 100 testing data points per task
            test_x = np.concatenate((test_x, x[indx[500:600], :]), axis=0)
            test_y = np.concatenate((test_y, y[indx[500:600]]), axis=0)

    return train_x1, train_y1, train_x2, train_y2, test_x, test_y


# Runs the experiments
def LF_experiment(
    angle, data_x, data_y, granularity, max_depth, reps=1, ntrees=29, acorn=None
):

    # Set random seed to acorn if acorn is specified
    if acorn is not None:
        np.random.seed(acorn)

    errors = np.zeros(
        2
    )  # initializes array of errors that will be generated during each rep

    for rep in range(reps):
        # training and testing subsets are randomly selected by calling the cross_val_data function
        train_x1, train_y1, train_x2, train_y2, test_x, test_y = cross_val_data(
            data_x, data_y, total_cls=10
        )

        # Change data angle for second task
        tmp_data = train_x2.copy()
        _tmp_ = np.zeros((32, 32, 3), dtype=int)
        total_data = tmp_data.shape[0]

        for i in range(total_data):
            tmp_ = image_aug(tmp_data[i], angle)
            # 2D image is flattened into a 1D array as random forests can only take in flattened images as inputs
            tmp_data[i] = tmp_

        # .shape gives the dimensions of each numpy array
        # .reshape gives a new shape to the numpy array without changing its data
        train_x1 = train_x1.reshape(
            (
                train_x1.shape[0],
                train_x1.shape[1] * train_x1.shape[2] * train_x1.shape[3],
            )
        )
        tmp_data = tmp_data.reshape(
            (
                tmp_data.shape[0],
                tmp_data.shape[1] * tmp_data.shape[2] * tmp_data.shape[3],
            )
        )
        test_x = test_x.reshape(
            (test_x.shape[0], test_x.shape[1] * test_x.shape[2] * test_x.shape[3])
        )
        # number of trees (estimators) to use is passed as an argument because the default is 100 estimators
        progressive_learner = LifelongClassificationForest(
            n_estimators=ntrees, default_max_depth=max_depth
        )

        # Add the original task
        progressive_learner.add_task(X=train_x1, y=train_y1)

        # Predict and get errors for original task
        llf_single_task = progressive_learner.predict(test_x, task_id=0)

        # Add the new transformer
        progressive_learner.add_transformer(X=tmp_data, y=train_y2)

        # Predict and get errors with the new transformer
        llf_task1 = progressive_learner.predict(test_x, task_id=0)

        errors[1] = errors[1] + (
            1 - np.mean(llf_task1 == test_y)
        )  # errors from transfer learning
        errors[0] = errors[0] + (
            1 - np.mean(llf_single_task == test_y)
        )  # errors from original task

    errors = (
        errors / reps
    )  # errors are averaged across all reps ==> more reps means more accurate errors
    
    # Average errors for original task and transfer learning are returned for the angle tested
    return errors


# Rotates the image by the given angle and zooms in to remove unnecessary white space at the corners
# Some image data is lost during rotation because of the zoom
def image_aug(pic, angle, centroid_x=23, centroid_y=23, win=16, scale=1.45):
    # Calculates scaled dimensions of image
    im_sz = int(np.floor(pic.shape[0] * scale))
    pic_ = np.uint8(np.zeros((im_sz, im_sz, 3), dtype=int))

    # Uses zoom function from scipy.ndimage to zoom into the image
    pic_[:, :, 0] = ndimage.zoom(pic[:, :, 0], scale)
    pic_[:, :, 1] = ndimage.zoom(pic[:, :, 1], scale)
    pic_[:, :, 2] = ndimage.zoom(pic[:, :, 2], scale)

    # Rotates image using rotate function from skimage.transform
    image_aug = rotate(pic_, angle, resize=False)
    image_aug_ = image_aug[
        centroid_x - win : centroid_x + win, centroid_y - win : centroid_y + win, :
    ]

    # Converts the image to unsigned byte format with values in [0, 255] and then returns it
    return img_as_ubyte(image_aug_)


def plot_bte(bte, angles):
    # Choose which color to make the results
    clr = ["#00008B"]
    c = sns.color_palette(clr, n_colors=1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot the data
    ax.plot(angles, bte, c=c[0], label="L2F", linewidth=3)

    # Format and label the plot
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.tick_params(labelsize=20)
    ax.set_xlabel("Angle of Rotation (Degrees)", fontsize=24)
    ax.set_ylabel("Backward Transfer Efficiency", fontsize=24)
    ax.set_title("Rotation Experiment", fontsize=24)
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)
    plt.tight_layout()
    # x.legend(fontsize = 24)
    plt.show()
