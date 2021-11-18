import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def color_distortion(image, s=.5):
    """
    From paper: A Simple Framework for Contrastive Learning of Visual Representations, Chen et al
    """
    # image is a tensor with value range in [0, 1].
    # s is the strength of color distortion.

    def color_jitter(x):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.8*s)
        x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_hue(x, max_delta=0.2*s)
        x = tf.clip_by_value(x, 0, 1)
        return x
    def color_drop(x):
        x = tf.image.rgb_to_grayscale(x)
        #image = tf.tile(x, [1, 1, 3])
        return x

    def random_apply(f, img, p=1.):
        """
        Added for implementation
        """
        p_rand = np.random.uniform()
        if p_rand <= p:
            return f(img)
        else:
            return img

    # randomly apply transformation with probability p.
    image = random_apply(color_jitter, image, p=.8)
    image = random_apply(color_drop, image, p=.2)
    return image

def get_aug_seq(img_h, img_w):
    """
    X: batch of images
    """
    crop_div = 2
    #color_dist_strength = 1.

    aug_seq = tf.keras.Sequential([
        layers.RandomCrop(img_h // crop_div, img_w // crop_div), #random size?
        layers.Resizing(img_h, img_w), #RandomZoom instead of crop and resize?
        layers.RandomFlip(mode='horizontal'),
        layers.Lambda(color_distortion),
        #RandomGaussianBlur(),
        ])
    return aug_seq

def g(h_i):
    """
    Projection head, maps representations to contrastive loss space
    MLP with one hidden layer
    """
    return z

def sim(z_i, z_j):
    return tf.matmul(z_i, z_j, transpose_b=True) / (tf.norm(z_i) * tf.norm(z_j))

def contrastive_loss(z, i, j):
    """
    Loss function for positive pair of examples
    """
    N = np.size(z, 0) #this will be 2N where N is batch size because z has augs?
    tau = 1
    num = tf.math.exp(sim(z[i], z[j]) / tau)
    den = 0.
    for k in range(N):
        if k != i:
            den += tf.math.exp(sim(z[i], z[k]) / tau)
    return -tf.math.log(num / den)

def unsupcon(X, N, tau):
    img_h = np.size(X, 1)
    img_w = np.size(X, 2)
    for batch in X:
        for k in range(N):
            t = get_aug_seq(img_h, img_w)
            t_prime = get_aug_seq(img_h, img_w)
