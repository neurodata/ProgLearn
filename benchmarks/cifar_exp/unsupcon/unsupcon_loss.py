"""
Issue #520
Dante Basile
"""

from inception_preprocessing import distorted_bounding_box_crop
import numpy as np
import pickle
#from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Activation, Dense, Lambda, RandomFlip, Resizing
from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.applications.resnet50 import preprocess_input

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, images, batch_size, shuffle=True):
    super().__init__()
    self.images = images
    self.batch_size = batch_size
    self.shuffle = shuffle
    key_array = []
    self.key_array = np.arange(self.images.shape[0], dtype=np.uint32)
    self.on_epoch_end()

  def __len__(self):
    return len(self.key_array)//self.batch_size

  def __getitem__(self, index):
    keys = self.key_array[index*self.batch_size:(index+1)*self.batch_size]
    x = np.asarray(self.images[keys], dtype=np.float32)
    return x

  def on_epoch_end(self):
    if self.shuffle:
      self.key_array = np.random.permutation(self.key_array)

def build_rn_enc():
    """
    Default ResNet50 with input layer (224, 224, 3)
    Default output layer has been removed
    New output layer is the average pooling layer (2048)
    """
    base_model = keras.applications.ResNet50(weights=None,
                                             include_top=True
                                            )
    f = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    f.compile()
    return f

def build_proj_head():
    """
    Projection head, maps representations to contrastive loss space
    MLP with one hidden layer
    Input layer (2048)
    Output layer (128)
    """
    g = Sequential()
    g.add(Dense(128, input_dim=2048)) # W_1
    g.add(Activation('relu')) # \sigma
    g.add(Dense(128)) # W_2
    return g

def distorted_bounding_box_crop_wrapper(image):
    """
    From paper:
        area: [.08, 1.]
        aspect ratio: [3/4, 4/3]
    """
    image, bbox = distorted_bounding_box_crop(image, tf.zeros((1, 0, 4)),
                                       aspect_ratio_range=(3/4, 4/3),
                                       area_range=(.08, 1.)
                                      )
    return image

def color_distortion(image, s=.8):
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
        image = tf.tile(x, [1, 1, 3])
        return image

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
    aug_seq = keras.Sequential([
        Lambda(distorted_bounding_box_crop_wrapper),
        Resizing(224, 224),
        RandomFlip(mode='horizontal'),
        Lambda(color_distortion), # if preprocess, ensure this can handle negatives
        #RandomGaussianBlur(),
        ])
    return aug_seq

def get_x_tilde(x_batch, N, img_h, img_w, rsz_hw=224):
    """
    Sample two augmentation functions t and t_prime to make x_tilde
    """
    x_t = np.zeros((2*N, rsz_hw, rsz_hw, 3))
    for k in range(N):
        t = get_aug_seq(img_h, img_w)
        t_p = get_aug_seq(img_h, img_w)
        x_t[2*k] = t(x_batch[k])
        x_t[2*k + 1] = t_p(x_batch[k])
    return x_t

def sim(z_i, z_j):
    """
    Compute the similarity of the given vector pair
    """
    return tf.tensordot(z_i, z_j, 1) / (tf.norm(z_i) * tf.norm(z_j))

def get_pairwise_sim(z, N):
    """
    use the similarity function to construct a matrix of pairwise similarities
    """
    s_l_l = []
    for i in range(2*N):
        s_l = []
        for j in range(2*N):
            s_l.append(sim(z[i], z[j]))
        s_l_l.append(tf.stack(s_l))
    return tf.stack(s_l_l)

def contrastive_loss(s, i, j, N, tau=1.):
    """
    Contrastive loss function for a positive pair of examples
    """
    num = tf.math.exp(s[i, j] / tau)
    den = tf.constant(0.)
    for k in range(2*N):
        if k != i:
            den += tf.math.exp(s[i, k] / tau)
    return -1 * tf.math.log(num / den)

def model_loss(z, N):
    s = get_pairwise_sim(z, N)
    L = tf.constant(0.)
    for k in range(N):
        L += contrastive_loss(s, 2*k, 2*k + 1, N) + contrastive_loss(s, 2*k + 1, 2*k, N)
    L /= 2*N
    return L

def unsupcon_learning(X_train, n_epochs=10, N=16):
    """
    X: training data
    N: batch size [256, 8192]
    """
    loss_batch = []
    loss_epoch = []
    img_h = np.size(X_train, 1)
    img_w = np.size(X_train, 2)
    f = build_rn_enc()
    generator = DataGenerator(X_train, N)
    n_batches = len(generator)
    optimizer = keras.optimizers.SGD(learning_rate=.2, momentum=.4, nesterov=True)
    loss_train = np.zeros(shape=(n_epochs,), dtype=np.float32)
    for epoch in range(n_epochs):
        g = build_proj_head()
        epoch_loss_avg = keras.metrics.Mean()
        for batch in range(n_batches):
            x_batch = generator[batch]
            x_t = get_x_tilde(x_batch, N, img_h, img_w)
            with tf.GradientTape(persistent=True) as tape:
                h = f(x_t, training=True)
                z = g(h, training=True)
                L = model_loss(z, N)
            print(f"batch loss L: {L}")
            loss_batch.append(L)
            f_grad = tape.gradient(L, f.trainable_variables)
            g_grad = tape.gradient(L, g.trainable_variables)
            optimizer.apply_gradients(zip(f_grad, f.trainable_variables))
            optimizer.apply_gradients(zip(g_grad, g.trainable_variables))
            epoch_loss_avg(L)
        generator.on_epoch_end()
        loss_train[epoch] = epoch_loss_avg.result()
        print(f"epoch loss avg: {epoch_loss_avg.result()}")
        loss_epoch.append(epoch_loss_avg.result())
    return f, loss_batch, loss_epoch

def main():
    """
    N: hardcoded in unsupcon_learning
    tau: hardcoded in contrastive_loss
    """
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    #X_train = preprocess_input(X_train)
    #X_test = preprocess_input(X_test)
    X_train = X_train / 255.
    #X_test = X_test / 255.
    X_train = X_train[:1000]
    #X_test = X_test[:1000]
    f, loss_batch, loss_epoch = unsupcon_learning(X_train)
    f.save("unsupcon_RN50.h5")
    with open(r"loss_batch.pickle", 'wb') as outfile:
        pickle.dump(list(map(tf.get_static_value, loss_batch)), outfile)
    with open(r"loss_epoch.pickle", 'wb') as outfile:
        pickle.dump(list(map(tf.get_static_value, loss_epoch)), outfile)

if __name__ == '__main__':
    main()
