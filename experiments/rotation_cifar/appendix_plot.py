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
