#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil 

import sys
sys.path.append("../../src/")
from lifelong_dnn import LifeLongDNN
from joblib import Parallel, delayed

#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#%%
