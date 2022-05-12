from tensorflow import keras
import tensorflow as tf


def dice(seg, gt):
    """
    Compute Dice Score between segmentation and groud truth
    """
    intersection = tf.reduce_sum(tf.cast(tf.equal(seg, gt), tf.float32))
    return intersection * 2.0 / tf.cast(tf.size(seg) + tf.size(gt), tf.float32)


def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    """
    Down Block of UNet
    """
    c = keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides, activation="relu"
    )(x)
    c = keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides, activation="relu"
    )(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p


def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    """
    Up Block of UNet
    """
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides, activation="relu"
    )(concat)
    c = keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides, activation="relu"
    )(c)
    return c


def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    """
    Bottleneck of UNet
    """
    c = keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides, activation="relu"
    )(x)
    c = keras.layers.Conv2D(
        filters, kernel_size, padding=padding, strides=strides, activation="relu"
    )(c)
    return c


def UNet():
    """
    UNet Neural network
    """
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))

    p0 = inputs
    c1, p1 = down_block(p0, f[0])
    c2, p2 = down_block(p1, f[1])
    c3, p3 = down_block(p2, f[2])
    c4, p4 = down_block(p3, f[3])

    bn = bottleneck(p4, f[4])

    u1 = up_block(bn, c4, f[3])  # 8 -> 16
    u2 = up_block(u1, c3, f[2])  # 16 -> 32
    u3 = up_block(u2, c2, f[1])  # 32 -> 64
    u4 = up_block(u3, c1, f[0])  # 64 -> 128

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model
