###########################################################
# Contributor: Amy van Ee
# Functions to run scene_segmentation_rf_isic_tutorial
# 2022
###########################################################


# ========================================================#
# import packages
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, feature, future
from skimage.metrics import adapted_rand_error, variation_of_information
from functools import partial


# ========================================================#
# FUNCTION I

# returns dice coefficient as a measure of segmentation accuracy
# Input(s):
#   true - the true annotations
#   test - the predicted annotations
# Output(s):
#   dice coefficient


def get_dice(true, test):
    return np.size(test[test == true]) * 2.0 / (np.size(true) + np.size(test))


# ========================================================#
# FUNCTION II

# performs scene segmentation using the trained classifier
# Input(s):
#   img - the original image
#   lbl - the true annotations
#   clf - trained classifier
# Output(s):
#   N/A


def perform_sceneseg(img, lbl, clf):

    # set up parameters for training
    sigma_min = 1
    sigma_max = 16
    features_func = partial(
        feature.multiscale_basic_features,
        intensity=True,
        edges=False,
        texture=True,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        channel_axis=-1,
    )

    # get features
    features = features_func(img)
    pred = future.predict_segmenter(features, clf)

    # correction so that the "normal" label for the predicted
    # array matches that of the true array (both "0")
    pred[pred == 1] = 0

    ### QUANTIFICATION ###

    # calculate error, precision, recall, splits, merges, dice
    error, precision, recall = adapted_rand_error(lbl, pred)
    splits, merges = variation_of_information(lbl, pred)
    dice = get_dice(lbl, pred)

    # print results
    print(f"Adapted Rand error: {error}")
    print(f"Adapted Rand precision: {precision}")
    print(f"Adapted Rand recall: {recall}")
    print(f"False Splits: {splits}")
    print(f"False Merges: {merges}")
    print(f"Dice Coefficient: {dice}")

    ### VISUALIZATION ###

    # plot
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 6))

    ax[0].imshow(segmentation.mark_boundaries(img, pred, mode="thick"))
    ax[0].set_title("Image")
    ax[1].imshow(lbl, cmap=plt.cm.gray)
    ax[1].set_title("True Annotation")
    ax[2].imshow(pred, cmap=plt.cm.gray)
    ax[2].set_title("Segmentation")
    fig.tight_layout()

    plt.show()


# ========================================================#
# FUNCTION III

# performs scene segmentation on multidimensional input
# Input(s):
#   classifier - trained classifier
#   features - ndarray of features, first dimension is number of features, others
#              are compatible with shape of image to segment
# Output(s):
#   output - ndarray of predictions


def predict_sceneseg(classifier, features, task_id):
    sh = features.shape
    if features.ndim > 2:
        features = features.reshape((-1, sh[-1]))
    predicted_labels = classifier.predict(features, task_id)
    output = predicted_labels.reshape(sh[:-1])
    return output
