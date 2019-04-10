import tensorflow as tf
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
import numpy as np
import scipy.io


class MyModel(tf.keras.Model):
    pass


def get_weights(vgg_layers, i):
  weights = vgg_layers[i][0][0][2][0][0]
  W = tf.constant(weights)
  return W


def get_bias(vgg_layers, i):
  bias = vgg_layers[i][0][0][2][0][1]
  # (1, out_channels) -> (out_channels,1)
  b = tf.constant(np.reshape(bias, bias.size))
  return b

vgg_layers = []

def conv_layer(layer_name, layer_num):
    w = get_weights(vgg_layers, layer_num)
    b = get_bias(vgg_layers, layer_num)

    filters = w[3]
    kernel_size = [w[0], w[1]]
    has_bias = False
    bias_initial = None
    if b:
        has_bias = True
        bias_initial = tf.keras.initializers.Constant(b)
    conv = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        use_bias=has_bias,
        bias_initializer=bias_initial,
        activation='relu'
    )
    return conv


def build_model(input_img, model_weights_file):
    _, h, w, d = input_img.shape
    intput = np.zeros((1, h, w, d))

    # load weights
    global vgg_layers
    vgg_rawnet = scipy.io.loadmat(model_weights_file)
    vgg_layers = vgg_rawnet['layers'][0]

    model = models.Sequential()
    model.add
    model.add(conv_layer("conv1_1", 0))

    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam,
    #     #? 可否自定义损失函数
    #     loss=)
    model.fit(intput)