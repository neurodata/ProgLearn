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
import keras

from tqdm import tqdm

import tensorflow as tf

import sys
sys.path.append("../../src")

from lifelong_dnn import LifeLongDNN

def homogenize_labels(a):
    u = np.unique(a)
    return np.array([np.where(u == i)[0][0] for i in a])

def cross_val_data(data_x, data_y, total_cls=10):
    x = data_x.copy()
    y = data_y.copy()
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]
    
    
    for i in range(total_cls):
        indx = idx[i]#np.roll(idx[i],(cv-1)*100)
        random.shuffle(indx)
        
        if i==0:
            train_x1 = x[indx[0:250],:]
            train_x2 = x[indx[250:500],:]
            train_y1 = y[indx[0:250]]
            train_y2 = y[indx[250:500]]
            
            test_x = x[indx[500:600],:]
            test_y = y[indx[500:600]]
        else:
            train_x1 = np.concatenate((train_x1, x[indx[0:250],:]), axis=0)
            train_x2 = np.concatenate((train_x2, x[indx[250:500],:]), axis=0)
            train_y1 = np.concatenate((train_y1, y[indx[0:250]]), axis=0)
            train_y2 = np.concatenate((train_y2, y[indx[250:500]]), axis=0)
            
            test_x = np.concatenate((test_x, x[indx[500:600],:]), axis=0)
            test_y = np.concatenate((test_y, y[indx[500:600]]), axis=0)
        
        
    return train_x1, train_y1, train_x2, train_y2, test_x, test_y 

def LF_experiment(data_x, data_y, angle, reps=1, ntrees=29, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
    
    errors = np.zeros(2)
    
    for rep in tqdm(range(reps)):
        train_x1, train_y1, train_x2, train_y2, test_x, test_y = cross_val_data(data_x, data_y, total_cls=10)
    
    
        #change data angle for second task
        tmp_data = train_x2.copy()
        _tmp_ = np.zeros((32,32,3), dtype=int)
        total_data = tmp_data.shape[0]
        
        for i in range(total_data):
            tmp_ = image_aug(tmp_data[i],angle)
        
            tmp_data[i] = tmp_
        

        lifelong_forest = LifeLongDNN()
        lifelong_forest.new_forest(train_x1, train_y1, n_estimators=ntrees)
        lifelong_forest.new_forest(tmp_data, train_y2, n_estimators=ntrees)
    
        llf_task1=lifelong_forest.predict(test_x, representation='all', decider=0)
        llf_single_task=lifelong_forest.predict(test_x, representation=0, decider=0)
    
        m = len(test_y)
        errors[1] = errors[1]+(1 - np.sum(llf_task1 == test_y)/m)
        errors[0] = errors[0]+(1 - np.sum(llf_single_task == test_y)/m)
    
    errors = errors/reps
    print(errors,train_x1.shape,train_x2.shape,test_x.shape,angle)
    with open('rotation_res_ln/'+'LF'+'_'+str(angle)+'.pickle', 'wb') as f:
        pickle.dump(errors, f)
        
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

angles = np.arange(0,360,5)
cvs = np.arange(1,7)
task_label = range(0,10)

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()

data_x = np.concatenate((X_train, X_test), axis=0)
data_y = np.concatenate((y_train[:,0], y_test[:,0]), axis=0)

class_idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]

data_x1 = np.zeros((6000,32,32,3), dtype=int)
data_y1 = np.zeros((6000,1), dtype=int)
ind = 0

for j in range(10):
    for i in range(600):
        data_x1[ind] = data_x[class_idx[j][i]]
        data_y1[ind] = data_y[class_idx[j][i]]
        ind += 1
        
data_y1 = homogenize_labels(np.ravel(data_y1))

[LF_experiment(data_x1, data_y1, angle, reps=4, ntrees=16, acorn=1) for angle in tqdm(angles)]

fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.plot(angles,mean_eta)

#ax.set_yticks([1.0, 1.05, 1.1])
ax.tick_params(labelsize=20)
ax.set_xlabel('Angle of Rotation(Degree)', fontsize=24)
ax.set_ylabel('Transfer Efficiency', fontsize=24)

plt.savefig('cifar-100-rotate.png')