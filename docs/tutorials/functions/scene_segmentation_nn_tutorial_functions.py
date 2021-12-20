"""
Neural network functions for scene segmentation. Based off of 

"""
from keras import layers
import numpy as np

from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, concatenate
from keras import Model

import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

def unet(input_layer, num_classes, num_input_neurons=64):
    conv1 = Conv2D(num_input_neurons, (3,3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(num_input_neurons, (3,3), activation='relu', padding='same')(conv1)
    max_pool1 = MaxPool2D((2,2))(conv1)

    conv2 = Conv2D(num_input_neurons * 2, (3,3), activation='relu', padding='same')(max_pool1)
    conv2 = Conv2D(num_input_neurons * 2, (3,3), activation = 'relu', padding='same')(conv2)
    max_pool2 = MaxPool2D((2,2))(conv2)

    conv3 = Conv2D(num_input_neurons * 4, (3,3), activation='relu', padding='same')(max_pool2)
    conv3 = Conv2D(num_input_neurons * 4, (3,3), activation='relu', padding='same')(conv3)
    max_pool3 = MaxPool2D((2,2))(conv3)


    conv4 = Conv2D(num_input_neurons * 8, (3,3), activation='relu', padding='same')(max_pool3)
    conv4 = Conv2D(num_input_neurons * 8, (3,3), activation='relu', padding='same')(conv4)
    max_pool4 = MaxPool2D((2,2))(conv4)


    conv5 = Conv2D(num_input_neurons * 16, (3,3), activation='relu', padding='same')(max_pool4)
    conv5 = Conv2D(num_input_neurons * 16, (3,3), activation='relu', padding='same')(conv5)
    up_conv1 = Conv2DTranspose(num_input_neurons * 8, (3,3),strides=(2,2), activation='relu', padding='same')(conv5)
    up_conv1 = concatenate([up_conv1, conv4])
    up_conv1 = Conv2D(num_input_neurons * 8, (3,3), activation='relu', padding='same')(up_conv1)
    up_conv1 = Conv2D(num_input_neurons * 8, (3,3), activation='relu', padding='same')(up_conv1)

    up_conv2 = Conv2DTranspose(num_input_neurons * 4, (3,3),strides=(2,2), activation='relu', padding='same')(up_conv1)
    up_conv2 = concatenate([up_conv2, conv3])
    up_conv2 = Conv2D(num_input_neurons * 4, (3,3), activation='relu', padding='same')(up_conv2)
    up_conv2 = Conv2D(num_input_neurons * 4, (3,3), activation='relu', padding='same')(up_conv2)

    up_conv3 = Conv2DTranspose(num_input_neurons * 2, (3,3),strides=(2,2), activation='relu', padding='same')(up_conv2)
    up_conv3 = concatenate([up_conv3, conv2])
    up_conv3 = Conv2D(num_input_neurons * 2, (3,3), activation='relu', padding='same')(up_conv3)
    up_conv3 = Conv2D(num_input_neurons * 2, (3,3), activation='relu', padding='same')(up_conv3)

    up_conv4 = Conv2DTranspose(num_input_neurons, (3,3),strides=(2,2), activation='relu', padding='same')(up_conv3)
    up_conv4 = concatenate([up_conv4, conv1])
    up_conv4 = Conv2D(num_input_neurons, (3,3), activation='relu', padding='same')(up_conv4)
    up_conv4 = Conv2D(num_input_neurons, (3,3), activation='relu', padding='same')(up_conv4)

    return Model(inputs = input_layer, outputs = Conv2D(num_classes, (1,1), padding='same', activation='softmax')(up_conv4))

def load_images(imgpaths, lblpaths):
    X = np.array([cv2.imread(imgpath) for imgpath in imgpaths], dtype = object)
    Y = np.array([cv2.imread(lblpath) for lblpath in lblpaths], dtype = object)
    return X,Y


def load_image(image, mask):
  input_image = tf.image.resize(image, (256, 256), method='nearest')
  input_image = (input_image - tf.reduce_min(input_image)) / (tf.reduce_max(input_image) - tf.reduce_min(input_image)) * 255
  input_mask = tf.image.resize(mask, (256, 256), method='nearest')
  input_mask = (input_mask - tf.reduce_min(input_mask)) / (tf.reduce_max(input_mask) - tf.reduce_min(input_mask)) * 252
  return input_image, input_mask

def dice(y_true, y_pred):
    intersection = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    return (intersection * 2.0) / tf.cast((tf.size(y_true) + tf.size(y_pred)), tf.float32)

def load_data(imgpath, lblpath):
    input_image, input_mask = load_images(imgpath, lblpath)
    X = np.zeros((input_image.shape[0], 256, 256, 3))
    Y = np.zeros((input_mask.shape[0], 256, 256, 3))
    
    for i, (x, y) in enumerate(zip(input_image, input_mask)):
        X[i], Y[i] = tf.cast(load_image(x,y), tf.int16)

    n_classes = len(np.unique(Y.reshape(-1, 3), axis = 0))
    print("{} classes found".format(n_classes))
    Y = relabel(Y, n_classes)
    return X, Y, n_classes

def relabel(Y_old, n_classes):
    Y = np.zeros((Y_old.shape[0], 256, 256, 1))

    # relabeling
    for i, j in zip(np.unique(Y_old.reshape(-1, 3), axis = 0), np.arange(n_classes)):
        where = (Y_old == i).all(axis=3)
        Y[where] = j
    
    return Y

def evaluate(model, X_test, Y_test):
    dices = np.zeros(len(X_test))
    accuracy = np.zeros(len(X_test))
    mut_info = np.zeros(len(X_test))
    for i in np.arange(len(X_test)):
        m = tf.keras.metrics.Accuracy()
        pred = model.predict(tf.convert_to_tensor([X_test[i]]))
        score = dice(Y_test[i], pred)
        plt.figure(figsize=(24, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(tf.cast(X_test[i], tf.int32))
        plt.title("Image")
        plt.subplot(1,4,2)
        plt.imshow(Y_test[i].reshape(256,256), cmap = "Set1")
        plt.title("Ground Truth Segmentation Mask")
        plt.subplot(1,4,3)
        plt.imshow(np.argmax(pred, axis=3).reshape(256,256), cmap= "Set1")
        plt.title("Predicted Mask (Dice: {:.2f})".format(score.numpy()))
        dices[i] = score.numpy()
        m.update_state(Y_test[i].flatten(), np.argmax(pred, axis=3).flatten())
        accuracy[i] = m.result().numpy()
        plt.subplot(1,4,4)
        hist, x_edge, y_edge = np.histogram2d(Y_test[i].ravel(), np.argmax(pred,axis=3).ravel(), bins=20)
        plt.imshow(hist.T, cmap='gray', origin="lower", extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]])
        plt.xlabel("Ground Truth Segmentation Mask Label Bins")
        plt.ylabel("Predicted Mask Label Bins")
        mut_info[i] = mutual_information(hist)
        plt.title("Join Probability (Mutual Information: {:.2f}".format(mut_info[i]))
    print("Mean Test Dice: {:.2f}".format(np.mean(dices)))
    print("Mean Test Accuracy: {:.2f}".format(np.mean(accuracy)))

def mutual_information(hist):
    pxy = hist / float(np.sum(hist))
    
    # probability of x
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    return np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0] / px_py[pxy > 0]))

def plot_metrics(history):
    epochs = history.epoch
    plt.figure(figsize=(22,5))

    plt.subplot(1,3,1)
    plt.plot(epochs, history.history['loss'], label="Training Loss")
    plt.plot(epochs, history.history['val_loss'], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Sparse Categorical Cross Entropy Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(epochs, history.history['accuracy'], label="Training Accuracy")
    plt.plot(epochs, history.history['val_accuracy'], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.legend()
    plt.subplot(1,3,3)
    plt.plot(epochs, history.history['dice'], label="Training Dice")
    plt.plot(epochs, history.history['val_dice'], label="Validation Dice")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title("Dice Coefficient vs. Epoch")
