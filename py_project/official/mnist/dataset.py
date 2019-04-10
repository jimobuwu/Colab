from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import tempfile
from six.moves import urllib
import gzip
import shutil # shell utilities
import numpy as np


def read32(bytestream):
    # '>' 大端
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


# magic | num_images | rows | cols
def check_image_file_header(filename):
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           f.name))
        if rows != 28 or cols != 28:
            raise ValueError(
                'Invalid MNIST file %s: Excepted 28x28 images, found %dx%d' % (f.name, rows, cols))


def check_label_file_header(filename):
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           f.name))


def download(dir, filename):
    filepath = os.path.join(dir, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(dir):
        tf.gfile.MakeDirs(dir)
    url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, \
        tf.gfile.Open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath


def dataset(directory, images_file, labels_file):
    images_file = download(directory, images_file)
    labels_file = download(directory, labels_file)
    check_image_file_header(images_file)
    check_label_file_header(labels_file)

    def decode_image(image):
        # decode_raw, 将字符串解析成图像对应的像素数组
        # tf.string -> [tf.uint8]
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        return image / 255.0

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)
        label = tf.reshape(label, [])
        return tf.cast(label, tf.int32)

    images = tf.data.FixedLengthRecordDataset(
        # map() , Maps `map_func` across the elements of this dataset.
        images_file, 28 * 28, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
        labels_file, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))


def train(directory):
    return dataset(directory, 'train-images-idx3-ubyte',
                   'train-labels-idx1-ubyte')


def test(directory):
    return dataset(directory, 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')