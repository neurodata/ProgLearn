import cv2
import numpy as np

# ========================================================#
# Contributor: Narayani Wagle
# load_images - loads images and labels from paths and stores in matrix
# Inputs:
#   dim - tuple, dimensions to load images as
#   flatten_imgs - boolean, True if 3-dimension images should be converted to 1 dimensional vector, False if 3-dimension structure to be preserved
# Outputs:
#   X - np - matrix containing images
#   Y - np - matrix containing labels


def load_images(imgpaths, lblpaths, dim, flatten_imgs=False):
    if flatten_imgs:
        X = np.array(
            [
                cv2.resize(cv2.imread(imgpath), dim, interpolation=cv2.INTER_NEAREST)
                .flatten()
                .transpose()
                for imgpath in imgpaths
            ]
        )
        Y = np.array(
            [
                cv2.resize(cv2.imread(lblpath), dim, interpolation=cv2.INTER_NEAREST)
                .flatten()
                .transpose()
                for lblpath in lblpaths
            ]
        )
    else:
        X = np.array(
            [
                cv2.resize(cv2.imread(imgpath), dim, interpolation=cv2.INTER_NEAREST)
                for imgpath in imgpaths
            ]
        )
        Y = np.array(
            [
                cv2.resize(cv2.imread(lblpath), dim, interpolation=cv2.INTER_NEAREST)
                for lblpath in lblpaths
            ]
        )
    return X, Y


# Contributor: Amy van Ee
# function that returns dice coefficient as a measure of segmentation accuracy
def get_dice(true, test):
    return np.size(test[test == true]) * 2.0 / (np.size(true) + np.size(test))
