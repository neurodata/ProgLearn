#%%
import matplotlib.pyplot as plt
import random
import pickle
from skimage.transform import rotate
from scipy import ndimage
from skimage.util import img_as_ubyte
from joblib import Parallel, delayed
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.ensemble.forest import _generate_sample_indices
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from itertools import product
import tensorflow.keras as keras
import tensorflow as tf

from numba import cuda

import sys
sys.path.append("../../src")

from lifelong_dnn import LifeLongDNN

#%%
def image_aug(pic, angle, centroid_x=23, centroid_y=23, win=16, scale=1.45):
    im_sz = int(np.floor(pic.shape[0]*scale))
    pic_ = np.uint8(np.zeros((im_sz,im_sz,3),dtype=int))
    
    pic_[:,:,0] = ndimage.zoom(pic[:,:,0],scale)
    
    pic_[:,:,1] = ndimage.zoom(pic[:,:,1],scale)
    pic_[:,:,2] = ndimage.zoom(pic[:,:,2],scale)
    
    image_aug = rotate(pic_, angle, resize=False)
    #print(image_aug.shape)
    image_aug_ = image_aug[centroid_x-win:centroid_x+win,centroid_y-win:centroid_y+win,:]
    
    return img_as_ubyte(image_aug_)

# %%
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()

fig, ax = plt.subplots(1,2, figsize=(8,4))
cif = image_aug(X_train[34,:,:,:],0)
rotated_cif = image_aug(cif,45)
ax[0].imshow(cif)
ax[1].imshow(rotated_cif)

ax[0].set_title('CIFAR image', fontsize=14)
ax[1].set_title('45$^\circ$ rotated CIFAR image', fontsize=14)

ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xticks([])
ax[1].set_yticks([])

right_side = ax[0].spines["right"]
right_side.set_visible(False)
top_side = ax[0].spines["top"]
top_side.set_visible(False)
left_side = ax[0].spines["left"]
left_side.set_visible(False)
bottom_side = ax[0].spines["bottom"]
bottom_side.set_visible(False)

right_side = ax[1].spines["right"]
right_side.set_visible(False)
top_side = ax[1].spines["top"]
top_side.set_visible(False)
left_side = ax[1].spines["left"]
left_side.set_visible(False)
bottom_side = ax[1].spines["bottom"]
bottom_side.set_visible(False)

plt.savefig("results/figs/rotated_cifar.pdf")
# %%
