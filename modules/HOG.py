'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import data, color

from PIL import Image


def get_sobel_kernel(ksize):
    if (ksize % 2 == 0) or (ksize < 1):
        raise ValueError("Kernel size must be a positive odd number")
    _base = np.arange(ksize) - ksize // 2
    a = np.broadcast_to(_base, (ksize, ksize))
    b = ksize // 2 - np.abs(a).T
    s = np.sign(a)
    return a + s * b


def get_gaussian_kernel(ksize=3, sigma=-1.0):
    ksigma = 0.15 * ksize + 0.35 if sigma <= 0 else sigma
    i, j = np.mgrid[0:ksize, 0:ksize] - (ksize - 1) // 2
    kernel = np.exp(-(i ** 2 + j ** 2) / (2 * ksigma ** 2))
    return kernel / kernel.sum()


def get_laplacian_of_gaussian_kernel(ksize=3, sigma=-1.0):
    ksigma = 0.15 * ksize + 0.35 if sigma <= 0 else sigma
    i, j = np.mgrid[0:ksize, 0:ksize] - (ksize - 1) // 2
    kernel = (i ** 2 + j ** 2 - 2 * ksigma ** 2) / (ksigma ** 4) * np.exp(-(i ** 2 + j ** 2) / (2 * ksigma ** 2))
    return kernel - kernel.mean()


def tf_kernel_prep_4d(kernel, n_channels):
    return np.tile(kernel, (n_channels, 1, 1, 1)).swapaxes(0, 2).swapaxes(1, 3)


def tf_kernel_prep_3d(kernel, n_channels):
    return np.tile(kernel, (n_channels, 1, 1)).swapaxes(0, 1).swapaxes(1, 2)


def tf_filter2d(batch, kernel, strides=(1, 1), padding='SAME'):
    n_ch = batch.shape[3].value
    tf_kernel = tf.constant(tf_kernel_prep_4d(kernel, n_ch))
    return tf.nn.depthwise_conv2d(batch, tf_kernel, [1, strides[0], strides[1], 1], padding=padding)


def tf_deriv(batch, ksize=3, padding='SAME'):
    try:
        n_ch = batch.shape[3].value
    except:
        n_ch = int(batch.get_shape()[3])
    gx = tf_kernel_prep_3d(np.array([[0, 0, 0],
                                     [-1, 0, 1],
                                     [0, 0, 0]]), n_ch)
    gy = tf_kernel_prep_3d(np.array([[0, -1, 0],
                                     [0, 0, 0],
                                     [0, 1, 0]]), n_ch)
    kernel = tf.constant(np.stack([gx, gy], axis=-1), name="DerivKernel", dtype=np.float32)
    return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], padding=padding, name="GradXY")


def tf_sobel(batch, ksize=3, padding='SAME'):
    n_ch = batch.shape[3].value
    gx = tf_kernel_prep_3d(get_sobel_kernel(ksize), n_ch)
    gy = tf_kernel_prep_3d(get_sobel_kernel(ksize).T, n_ch)
    kernel = tf.constant(np.stack([gx, gy], axis=-1), dtype=np.float32)
    return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], padding=padding)


def tf_sharr(batch, ksize=3, padding='SAME'):
    n_ch = batch.shape[3].value
    gx = tf_kernel_prep_3d([[-3, 0, 3],
                            [-10, 0, 10],
                            [-3, 0, 3]], n_ch)
    gy = tf_kernel_prep_3d([[-3, -10, -3],
                            [0, 0, 0],
                            [3, 10, 3]], n_ch)
    kernel = tf.constant(np.stack([gx, gy], axis=-1), dtype=np.float32)
    return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], padding=padding)


def tf_laplacian(batch, padding='SAME'):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=batch.dtype)
    return tf_filter2d(batch, kernel, padding=padding)


def tf_boxfilter(batch, ksize=3, padding='SAME'):
    kernel = np.ones((ksize, ksize), dtype=batch.dtype) / ksize ** 2
    return tf_filter2d(batch, kernel, padding=padding)


def tf_rad2deg(rad):
    return 180 * rad / tf.constant(np.pi)


def tf_select_by_idx(a, idx, grayscale):
    if grayscale:
        return a[:, :, :, 0]
    else:
        return tf.where(tf.equal(idx, 2),
                        a[:, :, :, 2],
                        tf.where(tf.equal(idx, 1),
                                 a[:, :, :, 1],
                                 a[:, :, :, 0]))


def tf_hog_descriptor(images, cell_size=8, block_size=2, block_stride=1, n_bins=9,
                      grayscale=False):
    # images = tf.expand_dims(images, 0)
    batch_size, height, width, depth = images.shape
    # height, width, depth = images.shape
    # batch_size = 1
    scale_factor = tf.constant(180 / n_bins, name="scale_factor", dtype=tf.float32)

    img = tf.constant(images, name="ImgBatch", dtype=tf.float32)

    if grayscale:
        img = tf.image.rgb_to_grayscale(img, name="ImgGray")

    # automatically padding height and width to valid size (multiples of cell size)
    if height % cell_size != 0 or width % cell_size != 0:
        height = height + (cell_size - (height % cell_size)) % cell_size
        width = width + (cell_size - (width % cell_size)) % cell_size
        img = tf.image.resize(img, (height, width))

    # gradients
    grad = tf_deriv(img)
    g_x = grad[:, :, :, 0::2]
    g_y = grad[:, :, :, 1::2]

    # masking unwanted gradients of edge pixels
    mask_depth = 1 if grayscale else depth
    g_x_mask = np.ones((batch_size, height, width, mask_depth))
    g_y_mask = np.ones((batch_size, height, width, mask_depth))
    g_x_mask[:, :, (0, -1)] = 0
    g_y_mask[:, (0, -1)] = 0
    g_x_mask = tf.constant(g_x_mask, dtype=tf.float32)
    g_y_mask = tf.constant(g_y_mask, dtype=tf.float32)

    g_x = g_x * g_x_mask
    g_y = g_y * g_y_mask

    # maximum norm gradient selection
    g_norm = tf.sqrt(tf.square(g_x) + tf.square(g_y), "GradNorm")

    if not grayscale and depth != 1:
        # maximum norm gradient selection
        idx = tf.argmax(g_norm, 3)
        g_norm = tf.expand_dims(tf_select_by_idx(g_norm, idx, grayscale), -1)
        g_x = tf.expand_dims(tf_select_by_idx(g_x, idx, grayscale), -1)
        g_y = tf.expand_dims(tf_select_by_idx(g_y, idx, grayscale), -1)

    g_dir = tf_rad2deg(tf.atan2(g_y, g_x)) % 180
    g_bin = tf.cast(g_dir / scale_factor, tf.int32)

    # cells partitioning
    cell_norm = tf.nn.space_to_depth(g_norm, cell_size, name="GradCells")
    cell_bins = tf.nn.space_to_depth(g_bin, cell_size, name="BinsCells")

    # cells histograms
    hist = list()
    zero = tf.zeros(cell_bins.get_shape())
    for i in range(n_bins):
        mask = tf.equal(cell_bins, tf.constant(i, name="%i" % i))
        hist.append(tf.reduce_mean(tf.where(mask, cell_norm, zero), 3))
    hist = tf.transpose(tf.stack(hist), [1, 2, 3, 0], name="Hist")

    # blocks partitioning
    block_hist = tf.image.extract_patches(hist,
                                          sizes=[1, block_size, block_size, 1],
                                          strides=[1, block_stride, block_stride, 1],
                                          rates=[1, 1, 1, 1],
                                          padding='VALID',
                                          name="BlockHist")

    # block normalization
    block_hist = tf.nn.l2_normalize(block_hist, 3, epsilon=1.0)

    # HOG descriptor
    hog_descriptor = tf.reshape(block_hist,
                                [int(block_hist.get_shape()[0]),
                                 int(block_hist.get_shape()[1]) * \
                                 int(block_hist.get_shape()[2]) * \
                                 int(block_hist.get_shape()[3])],
                                name='HOGDescriptor')

    return hog_descriptor, block_hist, hist