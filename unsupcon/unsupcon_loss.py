import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
#from tensorflow.keras.applications.resnet50 import preprocess_input

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

def get_aug_seq(img_h, img_w, crop_div=2):
    """
    X: batch of images
    """
    aug_seq = keras.Sequential([
        layers.RandomCrop(img_h // crop_div, img_w // crop_div), #random size?
        layers.Resizing(224, 224), #RandomZoom instead of crop and resize?
        layers.RandomFlip(mode='horizontal'),
        layers.Lambda(color_distortion), # TODO: if preprocess, ensure this can handle negatives
        #RandomGaussianBlur(),
        ])
    return aug_seq

# TODO: include this projection head if necessary
def g(h_i):
    """
    Projection head, maps representations to contrastive loss space
    MLP with one hidden layer
    """
    return z

def sim(z_i, z_j):
    return tf.tensordot(z_i, z_j, 1) / (tf.norm(z_i) * tf.norm(z_j))

def contrastive_loss(z, s, i, j, tau=1.):
    """
    Loss function for positive pair of examples
    """
    N_2 = np.size(z, 0) #this will be 2N where N is batch size because z has augs
    z_i = z[i]
    z_j = z[j]
    num = tf.math.exp(s[i, j] / tau)
    den = 0.
    for k in range(N_2):
        if k != i:
            den += tf.math.exp(s[i, k] / tau)
    return -tf.math.log(num / den)

def model_loss(l, N):
    L = 0.
    for k in range(N):
        L += l[2*k, 2*k + 1] + l[2*k + 1, 2*k]
    L /= 2*N
    return L

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, images, labels, batch_size=128, shuffle=True):
    super().__init__()
    self.images = images
    self.labels = labels
    self.batch\_size = batch_size
    self.shuffle = shuffle
    key_array = []
    self.key_array = np.arange(self.images.shape[0], dtype=np.uint32)
    self.on_epoch_end()

  def __len__(self):
    return len(self.key_array)//self.batch_size

  def __getitem__(self, index):
    keys = self.key_array[index*self.batch_size:(index+1)*self.batch_size]
    x = np.asarray(self.images[keys], dtype=np.float32)
    y = np.asarray(self.labels[keys], dtype=np.float32)
    return x, y

  def on_epoch_end(self):
    if self.shuffle:
      self.key_array = np.random.permutation(self.key_array)

def unsupcon_learning(X_train, y_train, n_classes=10, n_epochs=10, N=256):
    """
    X: training data
    N: batch size [256, 8192]
    """
    img_ct = np.size(X, 0)
    img_h = np.size(X, 1)
    img_w = np.size(X, 2)
    base_model = keras.applications.ResNet50(weights=None,
                                             include_top=False
                                            ) #input layer (224, 224, 3)
    o = base_model.output
    o = GlobalAveragePooling2D()(o)
    f = Model(inputs=base_model.input, outputs=o)
    generator = DataGenerator(images=X_train, labels=y_train, batch_size=N, shuffle=True)
    optimizer = keras.optimizers.Adam()
    loss_train = np.zeros(shape=(n_epochs,), dtype=np.float32)
    acc_train = np.zeros(shape=(n_epochs,), dtype=np.float32)
    loss_val = np.zeros(shape=(n_epochs,))
    acc_val = np.zeros(shape=(n_epochs,))
    n_batches = len(generator)
    for epoch in range(n_epochs):
        epoch_loss_avg = keras.metrics.Mean()
        epoch_acc_avg = keras.metrics.Mean()
        for batch in range(n_batches):
            x, y = generator[batch]
            with tf.GradientTape() as tape:
                x_t = tf.zeros((2*N, img_h, img_w))
                h = tf.zeros((2*N, n_avg_pool_weights)) # TODO: n_avg_pool_weights
                z = tf.zeros((2*N, n_avg_pool_weights)) # TODO: projection head
                s = tf.zeros((2*N, 2*N))
                l = tf.zeros((2*N, 2*N))
                for k in range(N):
                    t = get_aug_seq(img_h, img_w)
                    t_p = get_aug_seq(img_h, img_w)
                    x_t[2*k] = t(batch[k])
                    y_ = f(x_t[2*k], training=True)
                    h[2*k] = y_
                    #z[2*k] = g(h[2*k]) # TODO: projection head
                    z[2*k] = y_
                    x_t[2*k + 1] = t_p(batch[k])
                    y_p = f(x_t[2*k + 1], training=True)
                    h[2*k + 1] = y_p
                    #z[2*k + 1] = g(h[2*k + 1]) # TODO: projection head
                    z[2*k + 1] = y_p
                    loss = contrastive_loss(z, 2*k, 2*k + 1)
                for i in range(2*N):
                    for j in range(2*N):
                        s[i, j] = sim(z[i], z[j])
                for i in range(2*N):
                    for j in range(2*N):
                        l[i, j] = contrastive_loss(z, s, i, j, tau=1.)
                L = model_loss(l, N)
            grad = tape.gradient(L, f.trainable_variables)
            optimizer.apply_gradients(zip(grad, f.trainable_variables))
            epoch_loss_avg(L)
            #epoch_acc_avg(accuracy_score(y_true=y, y_pred=np.argmax(y_, axis=-1)))
        generator.on_epoch_end()
        loss_train[epoch] = epoch_loss_avg.result()
        #acc_train[epoch] = epoch_acc_avg.result()
        #y_ = f.predict(x_val) /#Validation predictions
        #loss_val[epoch] = model_loss(l, N).numpy()
        #acc_val[epoch] = accuracy_score(y_true=y_val, y_pred=np.argmax(y_, axis=-1))

def main():
    """
    N: hardcoded in unsupcon_learning
    tau: hardcoded in contrastive_loss
    """
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    #X_train = preprocess_input(X_train)
    #X_test = preprocess_input(X_test)
    X_train = tf.math.multiply(1./255, X_train)
    X_test = tf.math.multiply(1./255, X_test)
    unsupcon_learning(X_train)

if __name__ == '__main__':
    main()
